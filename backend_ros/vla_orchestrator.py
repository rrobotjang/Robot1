#!/usr/bin/env python3
# orchestrator_node.py
"""
Orchestrator FSM node for VLA system (Depth camera + Ultrasonic + MoveIt + Gripper)
- Loads FSM from YAML (fsm_config.yaml)
- Calls action functions mapped to modules (perception, motion, control)
- FastAPI voice endpoint for start/stop
- Publishes /vla/state
"""

from __future__ import annotations
import threading
import time
import yaml
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Bool, Float32

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Try to import wrappers; use lightweight mocks if unavailable
try:
    from vla_perception.vla_perception.pc_detector import detect_from_pointcloud, PcDetector
except Exception:
    detect_from_pointcloud = None
    PcDetector = None

try:
    from vla_motion.vla_motion.moveit_client import MoveItClient
except Exception:
    MoveItClient = None

try:
    from vla_control.vla_control.gripper_interface import GripperInterface
except Exception:
    GripperInterface = None


# ----------------------------
# Config / Defaults
# ----------------------------
FSM_CONFIG_PATH = os.path.join(os.getcwd(), "fsm_config.yaml")
CAM_TOPIC = "/camera/depth/color/points"
ULTRA_TOPIC = "/ultrasonic/distance"
EMERGENCY_TOPIC = "/vla/emergency_stop"
STATE_TOPIC = "/vla/state"
MAX_RETRIES = 3
DEFAULT_ULTRA_SLOW = 0.10  # meters
DEFAULT_ULTRA_STOP = 0.035  # meters

# Embedded default FSM (used when fsm_config.yaml not present)
DEFAULT_FSM = {
    "states": [
        "IDLE", "SEARCH_OBJECT", "SELECT_GRASP", "APPROACH_PREVIEW",
        "ULTRASONIC_SLOW_APPROACH", "GRASP", "VERIFY_GRASP", "LIFT",
        "SELECT_PLACE_SLOT", "PLACE_PREVIEW", "ULTRASONIC_PLACE_APPROACH",
        "RELEASE", "RETREAT", "ERROR", "EMERGENCY"
    ],
    "transitions": [
        {"from": "IDLE", "to": "SEARCH_OBJECT", "trigger": "start_pick"},
        {"from": "SEARCH_OBJECT", "to": "SELECT_GRASP", "trigger": "object_found"},
        {"from": "SELECT_GRASP", "to": "APPROACH_PREVIEW", "trigger": "grasp_selected"},
        {"from": "APPROACH_PREVIEW", "to": "ULTRASONIC_SLOW_APPROACH", "trigger": "approach_ready"},
        {"from": "ULTRASONIC_SLOW_APPROACH", "to": "GRASP", "trigger": "ultra_close"},
        {"from": "GRASP", "to": "VERIFY_GRASP", "trigger": "grasp_done"},
        {"from": "VERIFY_GRASP", "to": "LIFT", "trigger": "grasp_verified"},
        {"from": "VERIFY_GRASP", "to": "SELECT_GRASP", "trigger": "grasp_failed"},
        {"from": "LIFT", "to": "SELECT_PLACE_SLOT", "trigger": "lift_done"},
        {"from": "SELECT_PLACE_SLOT", "to": "PLACE_PREVIEW", "trigger": "slot_selected"},
        {"from": "PLACE_PREVIEW", "to": "ULTRASONIC_PLACE_APPROACH", "trigger": "place_approach_ready"},
        {"from": "ULTRASONIC_PLACE_APPROACH", "to": "RELEASE", "trigger": "ultra_place_close"},
        {"from": "RELEASE", "to": "RETREAT", "trigger": "released"},
        {"from": "RETREAT", "to": "IDLE", "trigger": "retreated"},
        # global
        {"from": "ANY", "to": "ERROR", "trigger": "fail"},
        {"from": "ANY", "to": "EMERGENCY", "trigger": "emergency"},
    ],
    "actions": {
        "SEARCH_OBJECT": "perception.search_objects",
        "SELECT_GRASP": "perception.select_best_grasp",
        "APPROACH_PREVIEW": "motion.plan_approach",
        "ULTRASONIC_SLOW_APPROACH": "motion.ultrasonic_guided_approach",
        "GRASP": "control.close_gripper_for_pick",
        "VERIFY_GRASP": "control.verify_grasp_with_ultrasonic",
        "LIFT": "motion.lift_after_grasp",
        "SELECT_PLACE_SLOT": "orchestrator.select_place_slot",
        "PLACE_PREVIEW": "motion.plan_place_approach",
        "ULTRASONIC_PLACE_APPROACH": "motion.ultrasonic_guided_place_approach",
        "RELEASE": "control.open_gripper_for_release",
        "RETREAT": "motion.retreat_after_place"
    }
}


# ----------------------------
# FastAPI input model
# ----------------------------
class VoiceCmd(BaseModel):
    text: str
    user: Optional[str] = None


# ----------------------------
# FSM runner
# ----------------------------
class FSM:
    def __init__(self, config: dict, action_map: Dict[str, Callable]):
        self.states = config.get("states", [])
        self.transitions = config.get("transitions", [])
        self.actions = config.get("actions", {})
        self.current = "IDLE"
        self.action_map = action_map  # mapping "perception.search_objects" -> callable
        self.lock = threading.Lock()

    def trigger(self, event: str) -> Tuple[bool, Optional[str]]:
        with self.lock:
            # find transition
            for tr in self.transitions:
                frm = tr["from"]
                to = tr["to"]
                trig = tr["trigger"]
                if trig != event:
                    continue
                if frm == "ANY" or frm == self.current:
                    prev = self.current
                    self.current = to
                    return True, prev
            return False, None

    def execute_current_action(self) -> bool:
        """
        Execute the action mapped to the current state, if any.
        Returns True on success (or no action), False on failure.
        """
        action_key = self.actions.get(self.current)
        if not action_key:
            return True  # no action for this state
        func = self.action_map.get(action_key)
        if not func:
            return False
        try:
            result = func()
            # action function should return True/False or (True/False, data)
            if isinstance(result, tuple):
                return bool(result[0])
            return bool(result)
        except Exception as e:
            # log handled by caller
            raise


# ----------------------------
# Orchestrator Node (rclpy)
# ----------------------------
class Orchestrator(Node):
    def __init__(self, fsm_cfg: dict):
        super().__init__("vla_orchestrator")
        self.get_logger().info("Orchestrator starting...")

        # subscriptions / publishers
        self.create_subscription(PointCloud2, CAM_TOPIC, self._pc_cb, 10)
        self.create_subscription(Float32, ULTRA_TOPIC, self._ultra_cb, 10)
        self.create_subscription(Bool, EMERGENCY_TOPIC, self._emergency_cb, 10)
        self._state_pub = self.create_publisher(String, STATE_TOPIC, 10)

        # internal state
        self._latest_pc: Optional[PointCloud2] = None
        self._ultra_distance: Optional[float] = None
        self._emergency: bool = False
        self._task_lock = threading.Lock()
        self._last_ultra_ts = 0.0

        # hardware wrappers
        self.moveit = MoveItClient(group_name="manipulator") if MoveItClient else None
        self.gripper = GripperInterface(cmd_topic="/robotis/gripper/command",
                                       state_topic="/robotis/gripper/state") if GripperInterface else None
        # perception helper (class or function)
        self.perception_node = PcDetector() if PcDetector else None

        # FSM action map - populate mapping from string keys to local methods
        action_map = {
            # perception
            "perception.search_objects": self._action_search_objects,
            "perception.select_best_grasp": self._action_select_best_grasp,
            # motion
            "motion.plan_approach": self._action_plan_approach,
            "motion.ultrasonic_guided_approach": self._action_ultrasonic_guided_approach,
            "motion.lift_after_grasp": self._action_lift_after_grasp,
            "motion.plan_place_approach": self._action_plan_place_approach,
            "motion.ultrasonic_guided_place_approach": self._action_ultrasonic_guided_place_approach,
            "motion.retreat_after_place": self._action_retreat_after_place,
            # control
            "control.close_gripper_for_pick": self._action_close_gripper,
            "control.verify_grasp_with_ultrasonic": self._action_verify_grasp,
            "control.open_gripper_for_release": self._action_open_gripper,
            # orchestrator internal
            "orchestrator.select_place_slot": self._action_select_place_slot,
        }

        self.fsm = FSM(fsm_cfg, action_map)
        self._publish_state("IDLE")

    # ------- ROS callbacks -------
    def _pc_cb(self, msg: PointCloud2):
        self._latest_pc = msg

    def _ultra_cb(self, msg: Float32):
        self._ultra_distance = float(msg.data)
        self._last_ultra_ts = time.time()

    def _emergency_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("Emergency triggered via topic!")
            self._emergency = True
            self._publish_state("EMERGENCY")
            if self.moveit:
                try:
                    self.moveit.stop()
                except Exception:
                    pass
        else:
            self.get_logger().info("Emergency cleared (software).")
            self._emergency = False
            self._publish_state("IDLE")

    # ------- helpers -------
    def _publish_state(self, state: str):
        msg = String()
        msg.data = state
        self._state_pub.publish(msg)
        self.get_logger().info(f"STATE -> {state}")

    def _ultra_get(self) -> Optional[float]:
        # stale: if not updated for 0.5s consider None
        if time.time() - self._last_ultra_ts > 0.5:
            return None
        return self._ultra_distance

    # ------- FSM action implementations -------
    # Each returns True on success, False on failure. They may set internal context.

    def _action_search_objects(self) -> Tuple[bool, Any]:
        """Use perception to produce candidate list. Store in self._candidates."""
        if self._latest_pc is None:
            self.get_logger().info("No pointcloud yet for detection.")
            return False, None
        # try function or class
        try:
            if detect_from_pointcloud:
                self._candidates = detect_from_pointcloud(self._latest_pc)
            elif self.perception_node:
                self._candidates = self.perception_node.detect(self._latest_pc)
            else:
                # mock: single candidate in front
                self._candidates = [{"id": "mock1", "pose": {"x": 0.5, "y": 0.0, "z": 0.02}, "score": 0.5}]
            if not self._candidates:
                self.get_logger().info("No candidates returned.")
                return False, None
            self.get_logger().info(f"Detected {len(self._candidates)} candidate(s).")
            return True, self._candidates
        except Exception as e:
            self.get_logger().error(f"search_objects error: {e}")
            return False, None

    def _action_select_best_grasp(self) -> Tuple[bool, Any]:
        # choose highest score
        try:
            cands = getattr(self, "_candidates", [])
            if not cands:
                return False, None
            cands = sorted(cands, key=lambda x: -x.get("score", 0.0))
            self._current_target = cands[0]
            self.get_logger().info(f"Selected candidate {self._current_target.get('id')}")
            return True, self._current_target
        except Exception as e:
            self.get_logger().error(f"select_best_grasp error: {e}")
            return False, None

    def _action_plan_approach(self) -> bool:
        """Plan coarse approach pose (use MoveIt)."""
        tgt = getattr(self, "_current_target", None)
        if not tgt:
            return False
        pose = dict(tgt["pose"])
        pose["z"] = pose.get("z", 0.0) + 0.12
        self._planned_approach = pose
        if self.moveit:
            ok = self.moveit.plan_and_execute(pose)
            if not ok:
                self.get_logger().warn("plan_approach failed.")
            return bool(ok)
        else:
            self.get_logger().info("Mock plan_approach (no MoveIt).")
            return True

    def _action_ultrasonic_guided_approach(self) -> bool:
        """Final slow approach guided by ultrasonic."""
        tgt = getattr(self, "_current_target", None)
        if not tgt:
            return False
        final = dict(tgt["pose"])
        # try guided small-step approach using ultrasonic
        slow_thresh = DEFAULT_ULTRA_SLOW
        stop_thresh = DEFAULT_ULTRA_STOP

        steps = 40
        step_m = 0.01  # 1cm steps
        for i in range(steps):
            if self._emergency:
                return False
            d = self._ultra_get()
            # if ultrasonic present and close enough -> stop
            if d is not None and d <= stop_thresh:
                self.get_logger().info(f"Ultrasonic close ({d:.3f}m): ready to grasp")
                return True
            # compute next pose (move down by step)
            next_pose = dict(final)
            next_pose["z"] = next_pose.get("z", 0.0) - (i + 1) * step_m
            if self.moveit:
                ok = self.moveit.plan_and_execute(next_pose)
                if not ok:
                    self.get_logger().warn("ultra step plan failed.")
                    return False
            else:
                time.sleep(0.05)
            time.sleep(0.02)
        # exhausted steps
        self.get_logger().warn("ultrasonic guided approach exhausted steps")
        return False

    def _action_close_gripper(self) -> bool:
        if self.gripper:
            ok = self.gripper.close(timeout=3.0)
            if not ok:
                self.get_logger().warn("Gripper close not confirmed.")
            return ok
        else:
            self.get_logger().info("Mock gripper close")
            return True

    def _action_verify_grasp(self) -> bool:
        # verify via ultrasonic (object presence) or other feedback
        expected = 0.06
        timeout = 1.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = self._ultra_get()
            if d is not None and d < expected:
                self.get_logger().info("Grasp verified by ultrasonic.")
                return True
            time.sleep(0.05)
        self.get_logger().warn("Grasp verification failed.")
        return False

    def _action_lift_after_grasp(self) -> bool:
        # simple lift
        tgt = getattr(self, "_current_target", None)
        if not tgt:
            return False
        lift_pose = dict(tgt["pose"])
        lift_pose["z"] = lift_pose.get("z", 0.0) + 0.20
        if self.moveit:
            return bool(self.moveit.plan_and_execute(lift_pose))
        else:
            time.sleep(0.1)
            return True

    def _action_select_place_slot(self) -> Tuple[bool, Any]:
        # placeholder: find slot (mock)
        slot = {"x": 0.8, "y": 0.2, "z": 0.45}
        self._place_slot = slot
        return True, slot

    def _action_plan_place_approach(self) -> bool:
        slot = getattr(self, "_place_slot", None)
        if not slot:
            return False
        approach = dict(slot)
        approach["z"] = approach.get("z", 0.0) + 0.12
        self._planned_place_approach = approach
        if self.moveit:
            return bool(self.moveit.plan_and_execute(approach))
        else:
            time.sleep(0.05)
            return True

    def _action_ultrasonic_guided_place_approach(self) -> bool:
        slot = getattr(self, "_place_slot", None)
        if not slot:
            return False
        final = dict(slot)
        stop_thresh = 0.035
        steps = 30
        step_m = 0.01
        for i in range(steps):
            if self._emergency:
                return False
            d = self._ultra_get()
            if d is not None and d <= stop_thresh:
                self.get_logger().info("Place approach confirmed by ultrasonic.")
                return True
            next_pose = dict(final)
            next_pose["z"] = next_pose.get("z", 0.0) - (i + 1) * step_m
            if self.moveit:
                ok = self.moveit.plan_and_execute(next_pose)
                if not ok:
                    return False
            else:
                time.sleep(0.02)
            time.sleep(0.02)
        return False

    def _action_open_gripper(self) -> bool:
        if self.gripper:
            ok = self.gripper.open(timeout=3.0)
            if not ok:
                self.get_logger().warn("Gripper open not confirmed.")
            return ok
        else:
            time.sleep(0.02)
            return True

    def _action_retreat_after_place(self) -> bool:
        # simple retreat to planned approach
        approach = getattr(self, "_planned_place_approach", None)
        if not approach:
            return False
        retreat = dict(approach)
        retreat["z"] = retreat.get("z", 0.0) + 0.15
        if self.moveit:
            return bool(self.moveit.plan_and_execute(retreat))
        else:
            time.sleep(0.05)
            return True

    # ------- High-level orchestration entry -------
    def start_pick_and_place_async(self):
        t = threading.Thread(target=self._pick_and_place_workflow, daemon=True)
        t.start()

    def _pick_and_place_workflow(self):
        if self._emergency:
            self.get_logger().warn("Emergency active - refuse to start workflow.")
            return
        with self._task_lock:
            self._publish_state("BUSY")
            # FSM sequence driven by triggers according to DEFAULT_FSM
            # 1) SEARCH_OBJECT
            ok, _ = self._run_state_action_with_retries("SEARCH_OBJECT")
            if not ok:
                self._publish_state("IDLE")
                return
            # fire transition
            self.fsm.trigger("object_found")

            # SELECT_GRASP
            ok, _ = self._run_state_action_with_retries("SELECT_GRASP")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("grasp_selected")

            # APPROACH_PREVIEW
            ok = self._run_state_action_with_retries("APPROACH_PREVIEW")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("approach_ready")

            # ULTRASONIC_SLOW_APPROACH
            ok = self._run_state_action_with_retries("ULTRASONIC_SLOW_APPROACH")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("ultra_close")

            # GRASP
            ok = self._run_state_action_with_retries("GRASP")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("grasp_done")

            # VERIFY_GRASP
            ok = self._run_state_action_with_retries("VERIFY_GRASP")
            if not ok:
                # retry flow
                attempts = 0
                while attempts < MAX_RETRIES and not ok:
                    attempts += 1
                    self.get_logger().info(f"Retry verify grasp {attempts}")
                    ok = self._run_state_action_with_retries("GRASP")
                    if ok:
                        ok = self._run_state_action_with_retries("VERIFY_GRASP")
                        if ok:
                            break
                if not ok:
                    self._publish_state("ERROR"); return
            self.fsm.trigger("grasp_verified")

            # LIFT
            ok = self._run_state_action_with_retries("LIFT")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("lift_done")

            # SELECT_PLACE_SLOT
            ok = self._run_state_action_with_retries("SELECT_PLACE_SLOT")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("slot_selected")

            # PLACE_PREVIEW
            ok = self._run_state_action_with_retries("PLACE_PREVIEW")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("place_approach_ready")

            # ULTRASONIC_PLACE_APPROACH
            ok = self._run_state_action_with_retries("ULTRASONIC_PLACE_APPROACH")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("ultra_place_close")

            # RELEASE
            ok = self._run_state_action_with_retries("RELEASE")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("released")

            # RETREAT
            ok = self._run_state_action_with_retries("RETREAT")
            if not ok:
                self._publish_state("ERROR"); return
            self.fsm.trigger("retreated")

            self._publish_state("IDLE")
            self.get_logger().info("Pick & Place workflow completed.")

    def _run_state_action_with_retries(self, state_name: str) -> bool:
        """Helper: set FSM current state, execute its action with retries."""
        # set fsm.current so action mapping uses it if necessary
        self.fsm.current = state_name
        tries = 0
        while tries < MAX_RETRIES and not self._emergency:
            tries += 1
            try:
                ok = self.fsm.execute_current_action()
            except Exception as e:
                self.get_logger().error(f"Action {state_name} exception: {e}")
                ok = False
            if ok:
                return True
            self.get_logger().warn(f"Action {state_name} failed on attempt {tries}")
            time.sleep(0.2)
        return False


# ----------------------------
# FastAPI wrapper / bootstrap
# ----------------------------
app = FastAPI()
node_singleton: Optional[Orchestrator] = None


@app.post("/voice")
def voice_endpoint(cmd: VoiceCmd):
    global node_singleton
    if node_singleton is None:
        return {"ok": False, "message": "Orchestrator not started yet"}
    txt = cmd.text.lower()
    if "정지" in txt or "멈춰" in txt or "stop" in txt:
        node_singleton.get_logger().info("Voice -> emergency stop")
        node_singleton._emergency = True
        try:
            if node_singleton.moveit:
                node_singleton.moveit.stop()
        except Exception:
            pass
        node_singleton._publish_state("EMERGENCY")
        return {"ok": True, "message": "Emergency triggered"}
    if "픽업" in txt or "pick" in txt:
        node_singleton.get_logger().info("Voice -> start pick workflow")
        node_singleton.start_pick_and_place_async()
        return {"ok": True, "message": "Pick & place started"}
    return {"ok": False, "message": "Unknown command"}


def ros_spin_thread(node: Orchestrator):
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"rclpy.spin exception: {e}")


def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


def load_fsm_config(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load FSM YAML {path}: {e}")
    return DEFAULT_FSM


def main():
    global node_singleton
    rclpy.init()
    cfg = load_fsm_config(FSM_CONFIG_PATH)
    node_singleton = Orchestrator(cfg)

    t_ros = threading.Thread(target=ros_spin_thread, args=(node_singleton,), daemon=True)
    t_ros.start()
    # small sleep to allow ROS spin up
    time.sleep(0.5)

    # Run FastAPI in main thread
    start_api()

    # shutdown
    try:
        node_singleton.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
