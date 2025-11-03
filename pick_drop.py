import cv2
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import math

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import DR_init
from dsr_example.simple.gripper_drl_controller import GripperController

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 50, 50

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def extract_instances_from_pcd(vtx,
                               voxel_size=0.004,
                               depth_axis='z',
                               ground_ratio=0.8,
                               dbscan_eps=0.03,
                               dbscan_min_samples=30,
                               global_min_pts=110,
                               local_density_r=0.015,
                               local_min_neighbors=20,
                               height_ratio=0.6):
    """포인트 클라우드에서 물체 인스턴스 추출"""
    if isinstance(vtx, np.ndarray):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
    else:
        pcd = vtx

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_denoised, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    points = np.asarray(pcd_denoised.points)
    
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[depth_axis]

    if len(points) == 0:
        return np.empty((0, 3)), np.array([]), []

    try:
        plane_model, inliers = pcd_denoised.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=1000)
        plane_pts = points[inliers]
        d_cut = np.max(plane_pts[:, axis_idx]) * ground_ratio
        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[inliers] = (plane_pts[:, axis_idx] >= d_cut)
    except:
        ground_mask = np.zeros(len(points), dtype=bool)

    labels_global = np.array(pcd_denoised.cluster_dbscan(eps=0.03, min_points=10))
    if np.any(labels_global >= 0):
        counts = np.bincount(labels_global[labels_global >= 0])
        keep = np.where(counts > global_min_pts)[0]
        sparse_mask = ~np.isin(labels_global, keep)
    else:
        sparse_mask = np.zeros(len(points), dtype=bool)

    tree = cKDTree(points)
    neighbor_counts = np.array([len(tree.query_ball_point(p, r=local_density_r)) for p in points])
    local_sparse_mask = neighbor_counts < local_min_neighbors
    depth_abnomaly = (points[:, 2] > 0.378) | (points[:, 2] < 0.15)
    mask_remove = ground_mask | sparse_mask | local_sparse_mask | depth_abnomaly
    filtered_points = points[~mask_remove]

    if len(filtered_points) == 0:
        return np.empty((0, 3)), np.array([]), []

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(filtered_points)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    results = []
    for lbl in range(num_clusters):
        cluster_pts = filtered_points[labels == lbl]
        if len(cluster_pts) < 10:
            continue

        center = np.mean(cluster_pts, axis=0)
        center[2] = np.max(cluster_pts[:, 2])

        
        other_axes = [i for i in range(3) if i != axis_idx]
        plane_pts = cluster_pts[:, other_axes]
        pca = PCA(n_components=2).fit(plane_pts)
        eigvals = pca.explained_variance_
        minor_idx = np.argmax(eigvals)
        main_dir = pca.components_[minor_idx]
        yaw = math.atan2(main_dir[1], main_dir[0])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw),  np.cos(-yaw)]])
        rotated = (cluster_pts[:, :2] - np.mean(cluster_pts[:, :2], axis=0)) @ R.T
        grip_length = rotated[:,1].max() - rotated[:,1].min()  # yaw 수직 방향 길이

        results.append({
            "id": lbl,
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
            "length":float(grip_length),
            "yaw": float(yaw),
            

        })

    return filtered_points, labels, results


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")

        self.bridge = CvBridge()
        self.get_logger().info("ROS 2 구독자 설정을 시작합니다...")

        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None

        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("컬러/뎁스/카메라정보 토픽 구독 대기 중...")

        # 그리퍼 초기화
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            wait(2)
            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed. Exiting.")
                raise Exception("Gripper initialization failed")
            self.get_logger().info("그리퍼를 활성화합니다...")
            self.gripper_is_open = True
            self.gripper.move(0)
            
        except Exception as e:
            self.get_logger().error(f"An error occurred during gripper setup: {e}")
            rclpy.shutdown()

        # 물체 탐지 관련 변수
        self.detected_objects = []
        self.auto_mode = False
        self.current_target_index = 0
        self.is_robot_moving = False

        self.get_logger().info("RealSense ROS 2 구독자와 로봇 컨트롤러가 초기화되었습니다.")

    def synced_callback(self, color_msg, depth_msg, info_msg):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge 변환 오류: {e}")
            return

        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            
            if info_msg.distortion_model == 'plumb_bob' or info_msg.distortion_model == 'rational_polynomial':
                self.intrinsics.model = rs.distortion.brown_conrady
            else:
                self.intrinsics.model = rs.distortion.none
            
            self.intrinsics.coeffs = list(info_msg.d)
            self.get_logger().info("카메라 내장 파라미터(Intrinsics) 수신 완료.")

    def detect_objects(self):
        """현재 프레임에서 물체 감지 및 중심점으로부터 거리 계산"""
        if self.latest_cv_depth_mm is None or self.intrinsics is None:
            self.get_logger().warn("뎁스 프레임 또는 카메라 정보가 없습니다.")
            return []
        
        # 로봇이 움직이는 동안에는 감지하지 않음
        if self.is_robot_moving:
            self.get_logger().warn("로봇 이동 중에는 물체를 감지하지 않습니다.")
            return []

        depth_scale = 0.001  # mm to m
        h, w = self.latest_cv_depth_mm.shape
        
        # 포인트 클라우드 생성
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        zs = self.latest_cv_depth_mm.astype(np.float32) * depth_scale
        
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy
        
        xs = (xs - cx) * zs / fx
        ys = (ys - cy) * zs / fy
        vtx = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)

        # 유효한 포인트만 필터링
        valid_mask = (zs.flatten() > 0.15) & (zs.flatten() < 0.378)
        if np.sum(valid_mask) < 5000:
            self.get_logger().warn("유효한 포인트가 너무 적습니다.")
            return []

        # 물체 감지
        _, _, instances = extract_instances_from_pcd(vtx, depth_axis='z')
        
        if len(instances) == 0:
            return []

        # 카메라 중심점(0, 0)으로부터의 거리 계산 (x, y 평면에서)
        center_x, center_y = 0, 0
        for inst in instances:
            dist_from_center = np.sqrt((inst["x"] - center_x)**2 + (inst["y"] - center_y)**2)
            inst["distance_from_center"] = dist_from_center
            
            # 2D 픽셀 좌표 계산
            if inst["z"] > 0:
                inst["pixel_u"] = int(inst["x"] * fx / inst["z"] + cx)
                inst["pixel_v"] = int(inst["y"] * fy / inst["z"] + cy)
            else:
                inst["pixel_u"] = -1
                inst["pixel_v"] = -1

        # 거리순으로 정렬 (가까운 순)
        instances_sorted = sorted(instances, key=lambda x: x["distance_from_center"])
        
        return instances_sorted

    def move_robot_and_control_gripper(self, cam_x, cam_y, cam_z, yaw, length):
        """로봇을 목표 위치로 이동하고 그리퍼 제어 (XYZ 이동)"""
        from DSR_ROBOT2 import movel, wait, movej, get_current_posj
        from DR_common2 import posx, posj
        
        # 상수 정의 (DR_common2에 없으므로 직접 정의)
        DR_TOOL = 1          # TOOL 좌표계
        DR_MV_MOD_REL = 1    # 상대 이동
        
        # 카메라-그리퍼 오프셋 (mm)
        GRIPPER_OFFSET_Y = -60   # Y축 -6cm
        GRIPPER_OFFSET_Z = -150  # Z축 -10cm (그리퍼가 카메라 아래에 있음)
        
        try:
            #그리퍼 min값에 따라 조절
            if length < 30:
                gripper_position = 680
                self.get_logger().info(f"작은 물체 감지 (크기: {length:.1f}mm) -> 그리퍼: {gripper_position}")
            else:
                gripper_position = 550
                self.get_logger().info(f"큰 물체 감지 (크기: {length:.1f}mm) -> 그리퍼: {gripper_position}")
            # ================================================
        
            # 그리퍼 열기
            self.get_logger().info("그리퍼 열기...")
            self.gripper.move(0)
            wait(2.0)
        

            # 로봇 이동 시작
            self.is_robot_moving = True
            
            # 카메라 좌표(m)를 로봇 좌표(mm)로 변환 + 오프셋 적용
            target_x = cam_x * 1000  # mm
            target_y = cam_y * 1000 + GRIPPER_OFFSET_Y  # mm (Y 오프셋)
            target_z = cam_z * 1000 + GRIPPER_OFFSET_Z  # mm (Z 오프셋, 뎁스 기반)

            # yaw를 degree로 변환
            yaw_deg = math.degrees(yaw) -100.0
            
            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"카메라 좌표 (m):    X={cam_x:.4f}, Y={cam_y:.4f}, Z={cam_z:.4f}")
            self.get_logger().info(f"물체 Yaw (rad/deg): {yaw:.4f} / {yaw_deg:.2f}°")
            self.get_logger().info(f"오프셋 적용 후 (mm): X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}")
            
          
            
            # 1단계: 물체 위치로 이동 (TOOL 좌표계 상대 이동)
            target_pos = posx(target_x, target_y, target_z, 0, 0, 0)
            
            self.get_logger().info("물체 위치로 이동 중...")
            movel(target_pos, vel=VELOCITY, acc=ACC, ref=DR_TOOL, mod=DR_MV_MOD_REL)
            wait(1.0)



             # 2단계: Joint 6을 yaw 각도로 회전
            self.get_logger().info(f"Joint 6을 {yaw_deg:.2f}도로 회전 중...")
            current_joints = get_current_posj()
            self.get_logger().info(f"현재 joint 위치: {current_joints}")
            
            # joint 6을 yaw 값으로 설정 (절대값)
            target_joints = posj(current_joints[0], current_joints[1], current_joints[2], 
                               current_joints[3], current_joints[4], yaw_deg)
            
            self.get_logger().info(f"목표 joint 6: {yaw_deg:.2f}°")
            movej(target_joints, vel=VELOCITY, acc=ACC)
            wait(1.5)
            
            #z축으로 살짝 이동
            movel(posx(0,0,50,0,0,0),vel=VELOCITY, acc=ACC, ref=DR_TOOL, mod=DR_MV_MOD_REL)
            wait(1.0)

            # 그리퍼 닫기
            self.get_logger().info(f"그리퍼 닫기 (위치: {gripper_position})...")
            self.gripper.move(gripper_position)
            wait(3.0)


            # 홈 위치로 복귀
            self.get_logger().info("=" * 60)
            self.get_logger().info("홈 위치로 복귀를 시작합니다...")


            
            try:
                from DSR_ROBOT2 import get_current_posj
                
                # 현재 위치 확인
                current_joint = get_current_posj()
                self.get_logger().info(f"현재 joint 위치: {current_joint}")
                
                # 홈 포즈 생성
                home_pos = posj(0, 0, 90, 0, 90, 0)
                self.get_logger().info(f"홈 포즈: {home_pos}")
                self.get_logger().info(f"홈 포즈 타입: {type(home_pos)}")
                
                # movej 실행
                self.get_logger().info("movej 명령 전송 시작...")
                result = movej(home_pos, vel=VELOCITY, acc=ACC)
                self.get_logger().info(f"movej 명령 결과: {result}")
                
                #지정 위치에 두기
                place_pos = posj(60, 0, 90, 0, 90, 0)
                movej(place_pos, vel=VELOCITY, acc=ACC)
                wait(1.0)

                 # 그리퍼 열기
                self.get_logger().info("그리퍼 열기...")
                self.gripper.move(0)
                wait(1.0)

                # movej 실행
                self.get_logger().info("movej 명령 전송 시작...")
                result = movej(home_pos, vel=VELOCITY, acc=ACC)
                self.get_logger().info(f"movej 명령 결과: {result}")


                # 충분한 대기
                self.get_logger().info("로봇 이동 대기 중...")
                wait(3.0)
                
                # 이동 후 위치 확인
                after_joint = get_current_posj()
                self.get_logger().info(f"이동 후 joint 위치: {after_joint}")
                self.get_logger().info("홈 위치 복귀 완료!")
                
            except Exception as home_error:
                import traceback
                self.get_logger().error(f"홈 복귀 중 오류: {home_error}")
                self.get_logger().error(f"상세 에러: {traceback.format_exc()}")

            self.get_logger().info("작업 완료!")
            self.get_logger().info(f"{'='*60}\n")
            
            # 로봇 이동 완료
            self.is_robot_moving = False

        except Exception as e:
            self.get_logger().error(f"로봇 제어 중 오류 발생: {e}")
            self.is_robot_moving = False

    def process_next_object(self):
        """다음 물체를 집으러 이동"""
        if self.current_target_index >= len(self.detected_objects):
            self.get_logger().info("\n" + "="*60)
            self.get_logger().info("모든 물체 처리 완료!")
            self.get_logger().info("="*60 + "\n")
            self.auto_mode = False
            self.current_target_index = 0  # 인덱스 초기화
            return

        obj = self.detected_objects[self.current_target_index]
        self.get_logger().info(f"\n타겟 {self.current_target_index + 1}/{len(self.detected_objects)}")
        self.get_logger().info(f"중심점으로부터 거리: {obj['distance_from_center']:.3f}m")

        # 로봇 이동 및 그리퍼 제어
        self.move_robot_and_control_gripper(obj['x'], obj['y'], obj['z'], obj['yaw'],obj['length'])
        
        # 다음 물체로 인덱스 증가
        self.current_target_index += 1

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()


def main(args=None):
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
    
        from DSR_ROBOT2 import movel, wait, movej, get_current_posj
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"DSR_ROBOT2 라이브러리를 임포트할 수 없습니다: {e}")
        rclpy.shutdown()
        exit(1)

    robot_controller = RobotControllerNode()

    cv2.namedWindow("RealSense Camera")

    print("\n" + "="*70)
    print("자동 물체 감지 및 로봇 제어 시스템 (XY 이동)")
    print("="*70)
    print("키 명령어:")
    print("  [SPACE] : 물체 감지 및 자동 수집 시작")
    print("  [ESC]   : 프로그램 종료")
    print("="*70)
    print("로봇 동작:")
    print("  1. 홈 포즈: posj(0, 0, 90, 0, 90, 0)")
    print("  2. TOOL 좌표계 기준 상대 이동 (ref=1, mod=1)")
    print("  3. 감지된 물체로 XYZ 이동 (뎁스 기반)")
    print("  4. 오프셋: Y=-60mm, Z=-150mm")
    print("  5. 작업 후 홈으로 복귀 → 다음 물체 자동 처리")
    print("="*70 + "\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)

            if robot_controller.latest_cv_color is not None:
                display_image = robot_controller.latest_cv_color.copy()
                
                h, w, _ = display_image.shape
                
                # 중심점 표시
                cv2.circle(display_image, (w // 2, h // 2), 5, (0, 0, 255), -1)
                cv2.line(display_image, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 0, 255), 2)
                cv2.line(display_image, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 0, 255), 2)
                
                # 감지된 물체 표시
                if len(robot_controller.detected_objects) > 0:
                    for idx, obj in enumerate(robot_controller.detected_objects):
                        u, v = obj["pixel_u"], obj["pixel_v"]
                        if 0 <= u < w and 0 <= v < h:
                            color = (0, 255, 0) if idx == robot_controller.current_target_index else (255, 255, 0)
                            cv2.rectangle(display_image, (u - 30, v - 30), (u + 30, v + 30), color, 2)
                            
                            # 물체 번호
                            cv2.putText(display_image, f"#{idx+1}", (u - 25, v - 35),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # 거리 정보
                            cv2.putText(display_image, f"{obj['distance_from_center']:.3f}m", (u - 25, v + 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # 카메라 좌표계 (미터)
                            cam_text = f"Cam: ({obj['x']:.3f}, {obj['y']:.3f}, {obj['z']:.3f})"
                            cv2.putText(display_image, cam_text, (u - 80, v + 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            # 로봇 목표 좌표계 (밀리미터) - 오프셋 적용
                            robot_x = obj['x'] * 1000
                            robot_y = obj['y'] * 1000 - 60   # Y축 -60mm
                            robot_z = obj['z'] * 1000 - 150  # Z축 -150mm (뎁스 기반)
                            robot_text = f"Rob: ({robot_x:.0f}, {robot_y:.0f}, {robot_z:.0f})"
                            cv2.putText(display_image, robot_text, (u - 80, v + 85),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 상태 정보 표시
                if robot_controller.is_robot_moving:
                    status_text = "ROBOT MOVING... (Camera paused)"
                    cv2.putText(display_image, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    status_text = "Standby (Press SPACE to detect)"
                    cv2.putText(display_image, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(robot_controller.detected_objects) > 0:
                    obj_text = f"Detected: {len(robot_controller.detected_objects)} objects"
                    cv2.putText(display_image, obj_text, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # 가장 가까운 물체 정보 추가
                    nearest_obj = robot_controller.detected_objects[0]
                    nearest_text = f"Nearest: {nearest_obj['distance_from_center']:.3f}m"
                    cv2.putText(display_image, nearest_text, (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("RealSense Camera", display_image)

            key = cv2.waitKey(1) & 0xFF
            
            # ESC 키: 종료
            if key == 27:
                break
            
            # SPACE 키: 물체 감지 및 자동 수집 시작
            elif key == 32:  # SPACE
                if not robot_controller.auto_mode:
                    print("\n물체 감지를 시작합니다...")
                    robot_controller.detected_objects = robot_controller.detect_objects()
                    
                    if len(robot_controller.detected_objects) > 0:
                        print(f"\n{'='*80}")
                        print(f"총 {len(robot_controller.detected_objects)}개의 물체가 감지되었습니다.")
                        print(f"{'='*80}")
                        print("\n중심점으로부터 가까운 순서:")
                        
                        for idx, obj in enumerate(robot_controller.detected_objects):
                            cam_x = obj['x']
                            cam_y = obj['y']
                            cam_z = obj['z']
                            distance = obj['distance_from_center']
                            length=obj['length']
                            
                            robot_x = cam_x * 1000
                            robot_y = cam_y * 1000 - 60   # Y축 -60mm
                            robot_z = cam_z * 1000 - 150  # Z축 -150mm (뎁스 기반)

                            gripper_pos = 650 if length < 30 else 520
                            
                            print(f"\n[물체 #{idx+1}]")
                            print(f"  중심점 거리:      {distance:.3f}m ({distance*1000:.1f}mm)")
                            print(f"  카메라 좌표 (m):  X={cam_x:.4f}, Y={cam_y:.4f}, Z={cam_z:.4f}")
                            print(f"  로봇 목표 (mm):   X={robot_x:.1f}, Y={robot_y:.1f}, Z={robot_z:.1f}")
                            print(f"  픽셀 좌표:        U={obj['pixel_u']}, V={obj['pixel_v']}")
                        
                        print(f"\n{'='*80}\n")
                        
                        # 자동 모드 활성화
                        robot_controller.auto_mode = True
                        robot_controller.current_target_index = 0
                        robot_controller.process_next_object()
                    else:
                        print("감지된 물체가 없습니다.")
            
            # 자동 모드에서 다음 물체 처리
            if robot_controller.auto_mode:
                if robot_controller.current_target_index < len(robot_controller.detected_objects):
                    time.sleep(0.5)  # 짧은 대기 후 다음 물체 처리
                    robot_controller.process_next_object()
                else:
                    # 모든 물체 처리 완료
                    robot_controller.auto_mode = False
                    robot_controller.current_target_index = 0
    
    except KeyboardInterrupt:
        print("\nCtrl+C로 종료합니다...")

    finally:
        print("프로그램을 종료합니다...")
        robot_controller.terminate_gripper()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        print("종료 완료.")

if __name__ == '__main__':
    main()
