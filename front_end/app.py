# main.py
"""
Monolithic prototype for:
- Depth camera -> PointCloud2 detection
- Doosan E0509 via MoveIt2 (planning & execution)
- Two-finger gripper control
- Place on shelf with occupancy check
- FastAPI endpoint for voice commands (execute/stop)
- Retry policy & emergency stop handling
"""

import threading
import time
import asyncio
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
# MoveIt2 Python APIs may differ; placeholder imports
# from moveit_commander import MoveGroupCommander

# -----------------------------
# Configuration / Constants
# -----------------------------
MAX_RETRIES = 3
SHELF_FRAME = "shelf_frame"
CAM_TOPIC = "/camera/depth/points"
EMERGENCY_TOPIC = "/vla/emergency_stop"
GRIPPER_CMD_TOPIC = "/vla/gripper/cmd"
GRIPPER_STATE_TOPIC = "/vla/gripper/state"
# MoveIt group names (adjust)
ARM_GROUP = "manipulator"
GRIPPER_GROUP = "gripper"

# -----------------------------
# ROS Node: Perception + Actuation
# -----------------------------
class VLAAgent(Node):
    def __init__(self):
        super().__init__("vla_agent_node")
        # subscribers
        self.create_subscription(PointCloud2, CAM_TOPIC, self.pc_callback, 10)
        self.emergency_sub = self.create_subscription(Bool, EMERGENCY_TOPIC, self.emergency_cb, 10)
        # gripper publisher
        self.gripper_pub = self.create_publisher(String, GRIPPER_CMD_TOPIC, 10)
        # state
        self.last_pc = None
        self.emergency_stop = False
        self.current_task_lock = threading.Lock()
        self.get_logger().info("VLAAgent initialized.")

        # TODO: initialize MoveIt/MoveGroup commander here
        # self.arm = MoveGroupCommander(ARM_GROUP)
        # self.gripper = MoveGroupCommander(GRIPPER_GROUP)

    def emergency_cb(self, msg: Bool):
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.get_logger().warn("Emergency stop received! Halting all motions.")
            # TODO: immediately stop MoveIt execution or send stop to robot controller
            # self.arm.stop()  # placeholder

    def pc_callback(self, pc_msg: PointCloud2):
        # store latest pointcloud (lightweight)
        self.last_pc = pc_msg

    # -----------------
    # Perception helpers
    # -----------------
    def detect_objects(self, pc_msg: PointCloud2) -> list:
        """
        Process pointcloud -> returns list of detected objects with pose and metadata.
        For brevity, returns mock objects: [{ 'id': 'can_1', 'pose': { 'x':..,'y':..,'z':..}, 'color': 'red'}]
        TODO: implement PCL plane removal, clustering, color filter (RGB -> from PointCloud or separate image)
        """
        # Placeholder detection:
        self.get_logger().info("Running mock detection on latest pointcloud.")
        # In real impl, convert PointCloud2 to numpy, do voxel filter, remove floor, cluster, compute centroids
        return [{"id":"can_1","pose":{'x':0.5,'y':0.1,'z':0.02}, "color":"red"}]

    # -----------------
    # Gripper helpers
    # -----------------
    def gripper_open(self):
        self.gripper_pub.publish(String(data="open"))
        # Wait/check state in real implementation
        time.sleep(0.5)

    def gripper_close(self):
        self.gripper_pub.publish(String(data="close"))
        time.sleep(0.5)

    # -----------------
    # Motion helpers
    # -----------------
    def plan_and_execute_pick(self, obj_pose: Dict[str,float]) -> bool:
        """
        Plan approach -> grasp -> lift using MoveIt and execute.
        Returns True if success.
        """
        if self.emergency_stop:
            self.get_logger().warn("Emergency active, aborting pick.")
            return False

        # TODO: compute approach pose, grasp pose in robot base frame using TF
        approach = obj_pose.copy()
        approach['z'] += 0.1  # approach from 10 cm above
        grasp = obj_pose.copy()
        grasp['z'] += 0.02  # close to surface

        self.get_logger().info(f"Planning approach to {approach} and grasp {grasp}.")
        # Placeholder: send planned joint trajectory to MoveIt
        try:
            # self.arm.set_pose_target(approach)
            # plan = self.arm.plan()
            # self.arm.execute(plan)
            # close gripper
            self.gripper_open()
            # move to grasp
            # self.arm.set_pose_target(grasp)
            # self.arm.go(wait=True)
            self.gripper_close()
            # lift
            # lift_pose = grasp.copy(); lift_pose['z'] += 0.15
            # self.arm.set_pose_target(lift_pose); self.arm.go(wait=True)
            self.get_logger().info("Mock pick executed.")
            return True
        except Exception as e:
            self.get_logger().error(f"Pick failed: {e}")
            return False

    def plan_and_place(self, target_pose: Dict[str,float]) -> bool:
        """
        Plan move to shelf cell and place object; uses shelf occupancy map (external).
        """
        if self.emergency_stop:
            self.get_logger().warn("Emergency active, aborting place.")
            return False

        try:
            # approach shelf cell
            # self.arm.set_pose_target(target_pose)
            # self.arm.go(wait=True)
            self.gripper_open()
            # retreat
            # retreat_pose = target_pose.copy(); retreat_pose['z'] += 0.1
            # self.arm.set_pose_target(retreat_pose); self.arm.go(wait=True)
            self.get_logger().info("Mock place executed.")
            return True
        except Exception as e:
            self.get_logger().error(f"Place failed: {e}")
            return False

    # -----------------
    # High-level flow
    # -----------------
    def pick_and_place_workflow(self):
        """
        1) Wait for pointcloud
        2) Detect objects
        3) For each object: try pick with retry, if failed 3 times -> alert
        4) Compute shelf target (empty location) via shelf_map service/calc
        5) Place and continue
        """
        if self.last_pc is None:
            self.get_logger().info("Waiting for initial pointcloud...")
            return

        objs = self.detect_objects(self.last_pc)
        for obj in objs:
            with self.current_task_lock:
                retries = 0
                success = False
                while retries < MAX_RETRIES and not success and not self.emergency_stop:
                    self.get_logger().info(f"Attempt {retries+1} to pick {obj['id']}")
                    success = self.plan_and_execute_pick(obj['pose'])
                    if not success:
                        retries += 1
                        time.sleep(0.5)  # small backoff
                if not success:
                    # alert: cannot pick
                    self.get_logger().error(f"Failed to pick {obj['id']} after {MAX_RETRIES} attempts. Raising alert.")
                    # In real system show popup / publish a topic for UI
                    # self.alert_pub.publish(String(data=f"FAILED_PICK:{obj['id']}"))
                    continue

                # find shelf placement target using shelf map (could be a service)
                target = self.find_shelf_slot()
                if target is None:
                    self.get_logger().error("No shelf slot available. Holding object and waiting.")
                    # optionally place to temporary holding area
                    continue

                placed = self.plan_and_place(target)
                if not placed:
                    self.get_logger().error("Place failed. Attempt recovery or manual intervention.")
                    # recovery logic here
                else:
                    self.get_logger().info(f"Placed {obj['id']} at {target}")

    def find_shelf_slot(self) -> Dict[str,float]:
        """
        Compute empty slot on shelf using shelf camera or internal occupancy map.
        For prototype returns a mock coordinate.
        """
        # TODO: implement occupancy map check using shelf camera pointcloud / projection
        return {'x':0.8,'y':0.2,'z':0.5}

# -----------------------------
# FastAPI: voice-command endpoints
# -----------------------------
api = FastAPI()
agent_node = None  # will be set after rclpy init

@api.post("/voice_command")
def voice_command_handler(cmd: Dict[str,Any]):
    """
    Expected payload: { 'text': '빨간 캔 픽업', 'user': 'operator1' }
    We'll implement a tiny rule-based parser here to create tasks.
    """
    text = cmd.get('text','').lower()
    if "정지" in text or "멈춰" in text:
        # emergency stop
        agent_node.emergency_cb(Bool(data=True))
        return {"ok": True, "message": "Emergency stop triggered."}
    # else map to simple 'pick' intent
    if "픽업" in text or "pick" in text:
        # For prototype just trigger pick & place sequence in background
        threading.Thread(target=agent_node.pick_and_place_workflow, daemon=True).start()
        return {"ok": True, "message": "Pick-and-place workflow started."}
    return {"ok": False, "message": "Unknown command."}

# -----------------------------
# Bootstrap: rclpy + FastAPI in same process (dev only)
# -----------------------------
def start_ros_node():
    global agent_node
    rclpy.init(args=None)
    agent_node = VLAAgent()
    try:
        rclpy.spin(agent_node)
    except Exception as e:
        print("ROS spin ended:", e)
    finally:
        agent_node.destroy_node()
        rclpy.shutdown()

def start_api():
    uvicorn.run(api, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start ROS in a thread and FastAPI in main thread (or vice versa)
    ros_thread = threading.Thread(target=start_ros_node, daemon=True)
    ros_thread.start()
    # give ros a moment
    time.sleep(1.0)
    start_api()

