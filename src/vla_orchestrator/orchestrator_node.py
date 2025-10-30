# orchestrator_node.py
"""
VLA Orchestrator Node (rclpy)
- PointCloud2 기반 detection (vla_perception)
- MoveIt2 planning/execution (vla_motion)
- Gripper control (vla_control)
- FastAPI endpoints for voice commands (start/stop)
"""

import threading
import time
from typing import Dict, Any, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
import numpy as np

# FastAPI for voice command interface
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Import local packages (these should be implemented in their packages)
# from vla_perception.pc_detector import detect_from_pointcloud
from vla_motion.moveit_client import MoveItClient
# from vla_control.gripper_interface import GripperInterface

# Configuration
CAM_TOPIC = "/camera/depth/color/points"    # adjust to your camera
EMERGENCY_TOPIC = "/vla/emergency_stop"
MAX_RETRIES = 3
SHELF_FRAME = "shelf_frame"

class VoiceCmd(BaseModel):
    text: str
    user: Optional[str] = None

class Orchestrator(Node):
    def __init__(self):
        super().__init__("vla_orchestrator")
        self.get_logger().info("Orchestrator starting...")

        # subscribe to pointcloud and emergency
        self.create_subscription(PointCloud2, CAM_TOPIC, self.pc_cb, 10)
        self.create_subscription(Bool, EMERGENCY_TOPIC, self.emergency_cb, 10)

        # internal state
        self.latest_pc: Optional[PointCloud2] = None
        self.emergency = False
        self.task_lock = threading.Lock()

        # instantiate hardware wrappers (replace with actual implementations)
        self.moveit = MoveItClient(group_name="manipulator")
        # self.gripper = GripperInterface(command_topic="/robotis/gripper/command")

    # --- callbacks ---
    def pc_cb(self, msg: PointCloud2):
        self.latest_pc = msg

    def emergency_cb(self, msg: Bool):
        self.emergency = bool(msg.data)
        if self.emergency:
            self.get_logger().warn("Emergency flag set. Halting activities.")
            # call stop on MoveIt or direct controller stop if available
            # self.moveit.stop()  # if available

    # --- high level utilities ---
    def compute_candidate_grasps(self, pc: PointCloud2) -> List[Dict[str, Any]]:
        """
        Converts PointCloud2 -> clusters -> grasp candidates.
        Here provide a placeholder returning mock candidate(s).
        Replace with real PCL/Open3D processing.
        """
        self.get_logger().info("Computing grasp candidates (mock)")
        # Return list of {id, pose(dict: x,y,z,rx,ry,rz), score}
        return [{"id":"obj_1", "pose":{"x":0.5,"y":0.0,"z":0.02,"rx":0,"ry":0,"rz":0}, "score":0.9}]

    def select_shelf_slot(self) -> Optional[Dict[str,float]]:
        """
        Determine an empty shelf slot using shelf camera / occupancy map.
        Return pose dict or None if no space.
        """
        # Placeholder: return a mock slot
        return {"x":0.8,"y":0.2,"z":0.45,"rx":0,"ry":0,"rz":0}

    def execute_pick(self, grasp_pose: Dict[str,float]) -> bool:
        if self.emergency:
            self.get_logger().warn("Emergency active — abort pick")
            return False

        # 1) approach pose (offset in z)
        approach = dict(grasp_pose)
        approach["z"] += 0.12

        # Plan + execute approach -> grasp -> lift using MoveIt
        try:
            self.get_logger().info(f"Plan approach to {approach}")
            self.moveit.set_pose_target(approach)
            plan_ok = self.moveit.plan_and_execute()
            # if not plan_ok: raise RuntimeError("Approach plan failed")
            # open gripper
            self.gripper.open()
            # plan to grasp
            self.get_logger().info(f"Plan grasp to {grasp_pose}")
            self.moveit.set_pose_target(grasp_pose)
            self.moveit.plan_and_execute()
            # close gripper
            self.gripper.close()
            # lift
            lift = dict(grasp_pose); lift["z"] += 0.2
            self.moveit.set_pose_target(lift); self.moveit.plan_and_execute()
            self.get_logger().info("Pick executed (mock)")
            return True
        except Exception as e:
            self.get_logger().error(f"Pick execution failed: {e}")
            return False

    def execute_place(self, slot_pose: Dict[str,float]) -> bool:
        if self.emergency:
            self.get_logger().warn("Emergency active — abort place")
            return False
        try:
            approach = dict(slot_pose); approach["z"] += 0.12
            self.moveit.set_pose_target(approach); self.moveit.plan_and_execute()
            self.moveit.set_pose_target(slot_pose); self.moveit.plan_and_execute()
            self.gripper.open()
            self.moveit.set_pose_target(approach); self.moveit.plan_and_execute()
            self.get_logger().info("Place executed (mock)")
            return True
        except Exception as e:
            self.get_logger().error(f"Place execution failed: {e}")
            return False

    # -------------------------
    # Main pick & place workflow
    # -------------------------
    def pick_and_place_workflow(self):
        if self.latest_pc is None:
            self.get_logger().info("No pointcloud yet — waiting")
            return

        with self.task_lock:
            # detect objects
            candidates = self.compute_candidate_grasps(self.latest_pc)
            if not candidates:
                self.get_logger().info("No candidates detected.")
                return

            # sort by score (best first)
            candidates = sorted(candidates, key=lambda c: -c["score"])

            for cand in candidates:
                obj_id = cand["id"]
                grasp_pose = cand["pose"]

                attempt = 0
                picked = False
                while attempt < MAX_RETRIES and not picked and not self.emergency:
                    self.get_logger().info(f"Picking {obj_id}, attempt {attempt+1}")
                    picked = self.execute_pick(grasp_pose)
                    if not picked:
                        attempt += 1
                        time.sleep(0.5)

                if not picked:
                    self.get_logger().error(f"Failed to pick {obj_id} after {MAX_RETRIES} attempts.")
                    # publish error topic or service for UI
                    continue

                # find shelf slot
                slot = self.select_shelf_slot()
                if slot is None:
                    self.get_logger().error("No shelf slot available. Holding object.")
                    # Optionally place in holding area or wait
                    continue

                placed = self.execute_place(slot)
                if not placed:
                    self.get_logger().error("Failed to place object. Recovery required.")
                else:
                    self.get_logger().info(f"Successfully placed {obj_id} at {slot}")

# --- FastAPI wrapper for voice commands (start/stop) ---
app = FastAPI()
orchestrator_node: Optional[Orchestrator] = None

@app.post("/voice")
def voice_endpoint(cmd: VoiceCmd):
    txt = cmd.text.lower()
    if "정지" in txt or "멈춰" in txt:
        # publish emergency topic
        orchestrator_node.get_logger().info("Voice -> emergency stop")
        orchestrator_node.emergency = True
        return {"ok": True, "message": "Emergency stop triggered"}
    if "픽업" in txt or "배치" in txt:
        threading.Thread(target=orchestrator_node.pick_and_place_workflow, daemon=True).start()
        return {"ok": True, "message": "Pick & place started"}
    return {"ok": False, "message": "Unknown command"}

# --- bootstrap ---
def ros_spin_thread():
    rclpy.init()
    global orchestrator_node
    orchestrator_node = Orchestrator()
    try:
        rclpy.spin(orchestrator_node)
    finally:
        orchestrator_node.destroy_node()
        rclpy.shutdown()

def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    t = threading.Thread(target=ros_spin_thread, daemon=True)
    t.start()
    time.sleep(1.0)
    start_api()
