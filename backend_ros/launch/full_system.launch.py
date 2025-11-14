from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        # =========================
        # RealSense depth camera
        # =========================
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            parameters=[{
                "pointcloud.enable": True,
                "align_depth.enable": True,
                "depth_module.profile": "640x480x30",
            }],
            output="screen"
        ),

        # =========================
        # Perception (PointCloud → object centroid)
        # =========================
        Node(
            package="vla_perception",
            executable="perception_node",
            parameters=[
                {"camera_topic": "/camera/depth/color/points"}
            ],
            output="screen"
        ),

        # =========================
        # Ultrasonic safety sensor
        # =========================
        Node(
            package="vla_control",
            executable="ultrasonic_node",
            parameters=[
                {"topic": "/ultrasonic/distance"},
                {"safe_distance": 0.15}
            ],
            output="screen"
        ),

        # =========================
        # Doosan Controller driver
        # =========================
        Node(
            package="doosan_robot_driver",
            executable="dsr_control_node",
            name="dsr_control",
            parameters=["/dsr_conf.yaml"],
            output="screen"
        ),

        # =========================
        # MoveIt2 motion planning
        # =========================
        Node(
            package="vla_motion",
            executable="motion_node",
            output="screen"
        ),

        # =========================
        # Main FSM / BehaviorTree orchestrator
        # =========================
        Node(
            package="vla_orchestrator",
            executable="orchestrator_node",
            parameters=[
                {"max_retry": 3},
                {"shelf_frame": "shelf_link"}
            ],
            output="screen"
        )
    ])
