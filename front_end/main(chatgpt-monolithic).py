# main.py
"""
Depth Camera 기반 객체 인식 → Doosan E0509 (MoveIt2) → Two-Finger Gripper Pick & Place
음성명령(API) 기반 동작 / 정지 제어 포함 (모놀리식 프로토타입)
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

MAX_RETRIES = 3
CAM_TOPIC = "/camera/depth/points"
EMERGENCY_TOPIC = "/vla/emergency_stop"
GRIPPER_CMD_TOPIC = "/vla/gripper/cmd"

class VLAAgent(Node):
    def __init__(self):
        super().__init__("vla_agent_node")
        self.create_subscription(PointCloud2, CAM_TOPIC, self.pc_callback, 10)
        self.create_subscription(Bool, EMERGENCY_TOPIC, self.emergency_cb, 10)
        self.gripper_pub = self.create_publisher(String, GRIPPER_CMD_TOPIC, 10)

        self.last_pc = None
        self.emergency_stop = False
        self.task_lock = threading.Lock()
        self.get_logger().info("VLAAgent initialized.")

    # --- ROS Callbacks ---
    def emergency_cb(self, msg: Bool):
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.get_logger().warn("Emergency Stop Triggered")

    def pc_callback(self, pc_msg: PointCloud2):
        self.last_pc = pc_msg

    # --- Perception ---
    def detect_objects(self, pc_msg: PointCloud2) -> list:
        """PointCloud 기반 객체 검출 (현재 mock 데이터 반환)"""
        self.get_logger().info("Detecting objects from PointCloud...")
        return [{"id": "can_1", "pose": {"x": 0.5, "y": 0.1, "z": 0.02}, "color": "red"}]

    # --- Gripper ---
    def gripper_open(self):
        self.gripper_pub.publish(String(data="open"))
        time.sleep(0.5)

    def gripper_close(self):
        self.gripper_pub.publish(String(data="close"))
        time.sleep(0.5)

    # --- Motion ---
    def plan_and_execute_pick(self, obj_pose: Dict[str, float]) -> bool:
        if self.emergency_stop:
            self.get_logger().warn("Emergency active, aborting pick.")
            return False
        try:
            self.gripper_open()
            self.get_logger().info(f"Picking object at {obj_pose}")
            self.gripper_close()
            self.get_logger().info("Pick successful")
            return True
        except Exception as e:
            self.get_logger().error(f"Pick failed: {e}")
            return False

    def plan_and_place(self, target_pose: Dict[str, float]) -> bool:
        if self.emergency_stop:
            self.get_logger().warn("Emergency active, aborting place.")
            return False
        try:
            self.get_logger().info(f"Placing object at {target_pose}")
            self.gripper_open()
            self.get_logger().info("Place successful")
            return True
        except Exception as e:
            self.get_logger().error(f"Place failed: {e}")
            return False

    # --- High-Level Logic ---
    def pick_and_place_workflow(self):
        if self.last_pc is None:
            self.get_logger().info("Waiting for PointCloud...")
            return

        objs = self.detect_objects(self.last_pc)
        for obj in objs:
            with self.task_lock:
                retries, success = 0, False
                while retries < MAX_RETRIES and not success and not self.emergency_stop:
                    self.get_logger().info(f"Attempt {retries + 1} to pick {obj['id']}")
                    success = self.plan_and_execute_pick(obj["pose"])
                    if not success:
                        retries += 1
                        time.sleep(0.5)

                if not success:
                    self.get_logger().error(f"Failed to pick {obj['id']} after {MAX_RETRIES} tries.")
                    continue

                target = self.find_shelf_slot()
                if not target:
                    self.get_logger().error("No shelf slot available, waiting.")
                    continue

                if self.plan_and_place(target):
                    self.get_logger().info(f"Placed {obj['id']} at {target}")
                else:
                    self.get_logger().error("Place failed.")

    def find_shelf_slot(self) -> Dict[str, float]:
        """빈 선반 좌표 계산 (현재 mock 값)"""
        return {"x": 0.8, "y": 0.2, "z": 0.5}

# --- FastAPI Interface ---
api = FastAPI()
agent_node = None

@api.post("/voice_command")
def voice_command_handler(cmd: Dict[str, Any]):
    text = cmd.get("text", "").lower()
    if "정지" in text or "멈춰" in text:
        agent_node.emergency_cb(Bool(data=True))
        return {"ok": True, "message": "Emergency Stop Activated"}
    if "픽업" in text or "pick" in text:
        threading.Thread(target=agent_node.pick_and_place_workflow, daemon=True).start()
        return {"ok": True, "message": "Pick & Place Started"}
    return {"ok": False, "message": "Unknown Command"}

# --- Execution ---
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
    ros_thread = threading.Thread(target=start_ros_node, daemon=True)
    ros_thread.start()
    time.sleep(1.0)
    start_api()
