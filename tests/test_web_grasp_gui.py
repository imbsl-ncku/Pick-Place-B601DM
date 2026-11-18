#!/usr/bin/env python3
"""
Unit and integration tests for web_grasp_gui.py.
Tests state management, REST endpoints, left-to-right sorting, and retreat waypoints.
"""

import copy
import json
import logging
import os
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tornado.testing
import yaml

from scripts.web_grasp_gui import (
    FIXED_GRASP_RPY,
    LOCAL_READY_POSE,
    RETREAT_WAYPOINTS,
    RobotWorker,
    SystemState,
    WebGuiLoggingHandler,
    _generate_synthetic_bench_frame,
    _generate_synthetic_tubes,
    make_app,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestWebGraspGuiLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg_path = PROJECT_ROOT / "config" / "default.yaml"
        self.waypoints_cfg_path = PROJECT_ROOT / "config" / "waypoints.yaml"
        self.state = SystemState(self.cfg_path, waypoints_cfg_path=self.waypoints_cfg_path, dry_run=True)

    def test_synthetic_tubes_left_to_right_sorting(self) -> None:
        """Synthetic tubes must be strictly sorted left-to-right by X-pixel coordinate."""
        tubes = _generate_synthetic_tubes(0.17)
        self.assertEqual(len(tubes), 4)

        # Check sorting by horizontal coordinate
        for i in range(len(tubes) - 1):
            self.assertLess(tubes[i]["center_px"][0], tubes[i + 1]["center_px"][0])

        # Assign IDs and verify 1, 2, 3, 4
        for idx, t in enumerate(tubes, 1):
            t["id"] = idx
        self.assertEqual([t["id"] for t in tubes], [1, 2, 3, 4])

    def test_synthetic_bench_frame_generation(self) -> None:
        """Synthetic frame must generate a valid 3-channel BGR numpy array."""
        frame = _generate_synthetic_bench_frame(640, 480)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_picking_queue_operations(self) -> None:
        """Enqueuing, dequeuing, and clearing must be thread-safe and maintain order."""
        tubes = _generate_synthetic_tubes(0.17)
        for idx, t in enumerate(tubes, 1):
            t["id"] = idx
        self.state.detected_tubes = tubes

        # Add tube #3, then tube #1 (custom order)
        ok1 = self.state.add_to_queue(3)
        ok2 = self.state.add_to_queue(1)
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(len(self.state.queue), 2)
        self.assertEqual(self.state.queue[0]["tube_id"], 3)
        self.assertEqual(self.state.queue[1]["tube_id"], 1)

        # Remove first item
        uid_to_remove = self.state.queue[0]["queue_uid"]
        self.state.remove_from_queue(uid_to_remove)
        self.assertEqual(len(self.state.queue), 1)
        self.assertEqual(self.state.queue[0]["tube_id"], 1)

        # Clear queue
        self.state.clear_queue()
        self.assertEqual(len(self.state.queue), 0)

    def test_exact_retreat_waypoints(self) -> None:
        """Verify strict adherence to RETREAT_WAYPOINTS without injected open_gripper calls."""
        self.assertGreaterEqual(len(self.state.retreat_waypoints), 2)
        # Waypoint 1: lift dz=0.10
        self.assertEqual(self.state.retreat_waypoints[0].get("dz"), 0.10)
        self.assertNotIn("gripper", self.state.retreat_waypoints[0])
        # Waypoint 2: tilt away with orientation
        self.assertEqual(self.state.retreat_waypoints[1].get("x"), 0.35)
        self.assertEqual(self.state.retreat_waypoints[1].get("y"), 0.30)
        self.assertEqual(self.state.retreat_waypoints[1].get("z"), 0.18)
        self.assertEqual(self.state.retreat_waypoints[1].get("roll"), -0.8)
        self.assertEqual(self.state.retreat_waypoints[1].get("pitch"), 0.0)
        self.assertEqual(self.state.retreat_waypoints[1].get("yaw"), 1.2)
        self.assertNotIn("gripper", self.state.retreat_waypoints[1])

    def test_waypoints_yaml_loading_and_saving(self) -> None:
        """Verify waypoints.yaml loading, modification, and persistence."""
        self.assertTrue(self.waypoints_cfg_path.exists())
        self.assertEqual(len(self.state.approach_waypoints), 0)
        self.assertGreaterEqual(len(self.state.retreat_waypoints), 2)

        # Test adding a temporary waypoint and saving to disk (with backup)
        orig_content = self.waypoints_cfg_path.read_text(encoding="utf-8")
        try:
            self.state.approach_waypoints.append({"x": 0.28, "y": 0.0, "z": 0.25, "duration": 2.0})
            ok = self.state.save_waypoints_to_disk()
            self.assertTrue(ok)

            # Reload from disk
            self.state.reload_waypoints_from_disk()
            self.assertEqual(len(self.state.approach_waypoints), 1)
            self.assertEqual(self.state.approach_waypoints[0]["z"], 0.25)
        finally:
            self.waypoints_cfg_path.write_text(orig_content, encoding="utf-8")
            self.state.reload_waypoints_from_disk()

    def test_log_levels_and_error_tracking(self) -> None:
        """Verify log recording with levels, error count tracking, and traceback capture."""
        self.state.clear_logs()
        self.assertEqual(len(self.state.logs), 1)  # "Telemetry logs cleared."
        self.assertEqual(self.state.error_count, 0)
        self.assertIsNone(self.state.last_error)

        # Log info, warning, success
        self.state.log("Information message", level="info")
        self.state.log_warning("Warning: Camera FPS drop")
        self.state.log_success("Motion completed")

        self.assertEqual(self.state.logs[-3]["level"], "info")
        self.assertEqual(self.state.logs[-2]["level"], "warning")
        self.assertEqual(self.state.logs[-1]["level"], "success")
        self.assertEqual(self.state.error_count, 0)

        # Log an error with exception
        try:
            raise ValueError("Test calibration matrix singular")
        except Exception as e:
            self.state.log_error(f"Computation failed: {e}", exc=e, source="Calibration")

        self.assertEqual(self.state.error_count, 1)
        self.assertIsNotNone(self.state.last_error)
        self.assertIn("Test calibration matrix singular", self.state.last_error["text"])
        self.assertIn("ValueError", self.state.last_error["traceback"])
        self.assertEqual(self.state.last_error["source"], "Calibration")

        # Clear last error
        self.state.clear_last_error()
        self.assertIsNone(self.state.last_error)
        self.assertEqual(self.state.error_count, 1)  # count remains for audit

        # Clear all logs
        self.state.clear_logs()
        self.assertEqual(self.state.error_count, 0)
        self.assertIsNone(self.state.last_error)

    def test_logging_handler_bridge(self) -> None:
        """Verify WebGuiLoggingHandler intercepts standard library warning and error logs."""
        bridge = WebGuiLoggingHandler(self.state)
        test_logger = logging.getLogger("TestDriverBridge")
        test_logger.addHandler(bridge)
        test_logger.setLevel(logging.DEBUG)

        try:
            initial_count = len(self.state.logs)
            test_logger.warning("Low battery detected on end effector")
            self.assertEqual(len(self.state.logs), initial_count + 1)
            self.assertEqual(self.state.logs[-1]["level"], "warning")
            self.assertIn("Low battery", self.state.logs[-1]["text"])

            test_logger.error("Motor bus communication lost")
            self.assertEqual(self.state.logs[-1]["level"], "error")
            self.assertEqual(self.state.error_count, 1)
            self.assertIn("Motor bus", self.state.last_error["text"])
        finally:
            test_logger.removeHandler(bridge)

    def test_robot_driver_feedback_enable_and_disable(self) -> None:
        """Robot enablement must depend on real hardware driver feedback, not just button press."""
        worker = RobotWorker(self.state)

        # 1. Hardware mock that succeeds
        mock_joint = SimpleNamespace(name="joint1")
        mock_motor = Mock(get_state=Mock(return_value=SimpleNamespace(pos=0.0, vel=0.0, torq=0.1)))
        mock_group = SimpleNamespace(_request_feedback=Mock(), _jcfgs=[mock_joint], _mm={"joint1": mock_motor})
        mock_arm_success = SimpleNamespace(
            enable_all=Mock(),
            disable_all=Mock(),
            groups={"arm": mock_group},
            joint_names=["joint1"],
            num_joints=1,
        )

        self.state.dry_run = False
        self.state.arm = mock_arm_success
        self.state.robot_enabled = False

        # Execute enable: hardware replies with valid state -> robot_enabled becomes True
        worker._execute_enable()
        self.assertTrue(self.state.robot_enabled)
        mock_arm_success.enable_all.assert_called_once()

        # Execute disable: -> robot_enabled becomes False
        worker._execute_disable()
        self.assertFalse(self.state.robot_enabled)
        mock_arm_success.disable_all.assert_called_once()

        # 2. Hardware mock that fails (motor unresponsive / returns None)
        mock_motor_dead = Mock(get_state=Mock(return_value=None))
        mock_group_dead = SimpleNamespace(_request_feedback=Mock(), _jcfgs=[mock_joint], _mm={"joint1": mock_motor_dead})
        mock_arm_dead = SimpleNamespace(
            enable_all=Mock(),
            disable_all=Mock(),
            groups={"arm": mock_group_dead},
            joint_names=["joint1"],
            num_joints=1,
        )

        self.state.arm = mock_arm_dead
        self.state.robot_enabled = False

        # Attempt to enable dead hardware: must NOT set robot_enabled = True!
        worker._execute_enable()
        self.assertFalse(self.state.robot_enabled)
        self.assertIsNotNone(self.state.last_error)
        self.assertIn("Motor feedback verification failed", self.state.last_error["text"])

    def test_disable_robot_performs_safe_home_first(self) -> None:
        """When disable is commanded on an enabled arm, it executes safe home first, then disables."""
        worker = RobotWorker(self.state)
        mock_ctrl = Mock(safe_home=Mock(), _running=True)
        mock_arm = Mock(disable_all=Mock())
        self.state.controller = mock_ctrl
        self.state.arm = mock_arm
        self.state.dry_run = False
        self.state.robot_enabled = True

        call_order = []
        mock_ctrl.safe_home.side_effect = lambda: call_order.append("safe_home")
        mock_arm.disable_all.side_effect = lambda: call_order.append("disable_all")

        worker._execute_disable()

        self.assertFalse(self.state.robot_enabled)
        self.assertEqual(call_order, ["safe_home", "disable_all"])
        self.assertTrue(any("Safe homing arm before disabling motors for safety" in log["text"] for log in self.state.logs))

    def test_emergency_stop_strict_order_execution(self) -> None:
        """Emergency stop must execute strictly in order: Stop -> Return Home (open gripper) -> Disable."""
        worker = RobotWorker(self.state)
        mock_ctrl = Mock(safe_home=Mock(), _running=True)
        mock_driver = Mock(open_gripper=Mock())
        mock_arm = Mock(disable_all=Mock())
        self.state.controller = mock_ctrl
        self.state.grasp_driver = mock_driver
        self.state.arm = mock_arm
        self.state.dry_run = False
        self.state.robot_enabled = True
        self.state.gripper_ready_open_m = 0.055

        call_order = []
        mock_driver.open_gripper.side_effect = lambda **kwargs: call_order.append(f"open_gripper({kwargs.get('distance_m', 0.055)})")
        mock_ctrl.safe_home.side_effect = lambda: call_order.append("safe_home")
        mock_arm.disable_all.side_effect = lambda: call_order.append("disable_all")

        # Set abort event to simulate execution being interrupted
        worker.abort_event.set()
        worker._execute_emergency_stop()

        self.assertFalse(self.state.robot_enabled)
        self.assertFalse(worker.abort_event.is_set())
        self.assertEqual(call_order, ["open_gripper(0.055)", "safe_home", "disable_all"])
        self.assertTrue(any("Step 1/3: Execution stopped immediately" in log["text"] for log in self.state.logs))
        self.assertTrue(any("Step 2/3: Opening gripper to Ready Open Width" in log["text"] for log in self.state.logs))
        self.assertTrue(any("Step 3/3: Disabling robot motors" in log["text"] for log in self.state.logs))

    def test_trigger_emergency_stop_aborts_active_execution(self) -> None:
        """trigger_emergency_stop sets abort_event, clears queues, signals controller and enqueues emergency_stop."""
        worker = RobotWorker(self.state)
        mock_ctrl = Mock(_stop_send=threading.Event())
        mock_driver = Mock(_state_lock=threading.Lock(), _grasp_result=None, _position_reached=False)
        self.state.controller = mock_ctrl
        self.state.grasp_driver = mock_driver
        self.state.queue = [{"queue_uid": 1, "tube_id": 1}]
        worker.command_queue.append(("grasp_queue", None))

        worker.trigger_emergency_stop()

        self.assertTrue(worker.abort_event.is_set())
        self.assertTrue(worker.state.pause_event.is_set())
        self.assertTrue(mock_ctrl._stop_send.is_set())
        self.assertEqual(len(self.state.queue), 0)
        self.assertEqual(list(worker.command_queue), [("emergency_stop", None)])

    def test_robot_worker_shutdown_disables_motors(self) -> None:
        """When terminal shutdown is invoked (Ctrl+C / interrupt), robot motors must be disabled and disconnected."""
        worker = RobotWorker(self.state)
        mock_ctrl = Mock(_running=True)
        mock_driver = Mock(stop=Mock())
        mock_arm = Mock(disable_all=Mock(), disconnect=Mock())
        self.state.controller = mock_ctrl
        self.state.grasp_driver = mock_driver
        self.state.arm = mock_arm
        self.state.dry_run = False
        self.state.robot_enabled = True
        self.state.robot_connected = True

        worker.shutdown()

        self.assertFalse(self.state.robot_enabled)
        self.assertFalse(self.state.robot_connected)
        mock_arm.disable_all.assert_called_once()
        mock_arm.disconnect.assert_called_once()
        self.assertTrue(any("Robot hardware confirmed DISABLED on shutdown" in log["text"] for log in self.state.logs))

    def test_robot_worker_blocks_execution_when_disabled(self) -> None:
        """RobotWorker must block motion execution and log short warning if robot_enabled is False."""
        worker = RobotWorker(self.state)
        self.state.robot_enabled = False
        self.state.queue = [{"queue_uid": 1, "label": "Tube #1", "p_base": np.zeros(3), "jaw_width_m": 0.03}]

        # Test _execute_queue
        worker._execute_queue()
        self.assertTrue(any("Robot disabled. Enable robot first." in log["text"] for log in self.state.logs))

        # Test _execute_macro_center
        worker._execute_macro_center(1)
        self.assertTrue(any("Robot disabled. Enable robot first." in log["text"] for log in self.state.logs))

        # Test _execute_safe_home
        worker._execute_safe_home()
        self.assertTrue(any("Robot disabled. Enable robot first." in log["text"] for log in self.state.logs))

        # Test _execute_ready_pose
        worker._execute_ready_pose()
        self.assertTrue(any("Robot disabled. Enable robot first." in log["text"] for log in self.state.logs))

    def test_robot_worker_execute_queue_applies_grasp_z_offset(self) -> None:
        """RobotWorker._execute_queue applies grasp_z_offset_m to the target Z coordinate."""
        worker = RobotWorker(self.state)
        self.state.robot_enabled = True
        self.state.dry_run = True
        self.state.cfg.setdefault("grasp_pipeline", {}).setdefault("grasp", {})["grasp_z_offset_m"] = -0.035
        p_base = np.array([0.35, 0.05, 0.17], dtype=np.float64)
        self.state.queue = [{"queue_uid": 1, "label": "Falcon Tube #1", "p_base": p_base, "jaw_width_m": 0.03}]

        worker._execute_queue()
        expected_log = "target=(0.350, 0.050, 0.135) [z_offset=-3.5cm]"
        self.assertTrue(any(expected_log in log["text"] for log in self.state.logs))


class TestWebGraspGuiApi(tornado.testing.AsyncHTTPTestCase):
    def get_app(self) -> tornado.web.Application:
        self.cfg_path = PROJECT_ROOT / "config" / "default.yaml"
        self.state = SystemState(self.cfg_path, dry_run=True)
        # Populate tubes
        tubes = _generate_synthetic_tubes(0.17)
        for idx, t in enumerate(tubes, 1):
            t["id"] = idx
        self.state.detected_tubes = tubes

        self.robot_worker = RobotWorker(self.state)
        # We don't start robot_worker thread to avoid background hardware commands during unit tests
        return make_app(self.state, self.robot_worker)

    def test_get_index_html(self) -> None:
        """GET / must return 200 with HTML containing key UI elements."""
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        body = response.body.decode("utf-8")
        self.assertIn("reBot Arm B601", body)
        self.assertIn("Visual Grasping", body)
        self.assertIn("Robot & Camera Config", body)
        self.assertIn("User Guide", body)
        self.assertIn("HOW TO PICK TUBES IN 3 STEPS", body)
        self.assertIn("GRASP QUEUE", body)
        self.assertIn("Macro Center", body)
        self.assertIn("Safe Home", body)
        self.assertIn("docs/WEB_GUI_USER_GUIDE.md", body)
        self.assertIn("led-indicator-bulb", body)
        self.assertIn("badge-enabled-container", body)
        self.assertIn("card-led-bulb", body)
        self.assertIn("Motor Driver Feedback", body)
        self.assertIn("cfg-grasp-z-offset", body)
        self.assertIn("tubes-bar", body)
        self.assertIn("lastTubesBarHtml", body)
        self.assertIn("btn-stop", body)
        self.assertIn("EMERGENCY STOP", body)
        self.assertIn("emergencyStop()", body)

    def test_get_status_api(self) -> None:
        """GET /api/status must return operational status, tube list, and queue."""
        response = self.fetch("/api/status")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertEqual(data["status"], "live")
        self.assertTrue(data["dry_run"])
        self.assertIn("robot_enabled", data)
        self.assertEqual(len(data["tubes"]), 4)
        self.assertEqual(data["tubes"][0]["id"], 1)
        self.assertEqual(data["tubes"][1]["id"], 2)

    def test_queue_api_workflow(self) -> None:
        """POST /api/queue add, remove, and clear endpoints."""
        # Add tube #2
        res_add = self.fetch(
            "/api/queue",
            method="POST",
            body=json.dumps({"action": "add", "tube_id": 2}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_add.code, 200)
        self.assertTrue(json.loads(res_add.body.decode("utf-8"))["success"])
        self.assertEqual(len(self.state.queue), 1)
        self.assertEqual(self.state.queue[0]["tube_id"], 2)

        # Remove tube #2
        queue_uid = self.state.queue[0]["queue_uid"]
        res_rem = self.fetch(
            "/api/queue",
            method="POST",
            body=json.dumps({"action": "remove", "queue_uid": queue_uid}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_rem.code, 200)
        self.assertEqual(len(self.state.queue), 0)

    def test_action_api_endpoints(self) -> None:
        """POST /api/action handles unfreeze, macro_center, and grasp."""
        # Unfreeze
        res = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "unfreeze"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.code, 200)
        self.assertEqual(self.state.status, "live")

        # Resume
        res = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "resume"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.code, 200)

        # Enable state so motion actions are permitted
        self.state.robot_enabled = True

        # Safe Home
        res_sh = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "safe_home"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_sh.code, 200)
        self.assertTrue(json.loads(res_sh.body.decode("utf-8"))["success"])

        # Ready Pose
        res_rp = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "ready_pose"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_rp.code, 200)
        self.assertTrue(json.loads(res_rp.body.decode("utf-8"))["success"])

        # Enable Robot API endpoint
        res_en = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "enable_robot"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_en.code, 200)
        self.assertTrue(json.loads(res_en.body.decode("utf-8"))["success"])

        # Disable Robot API endpoint
        res_dis = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "disable_robot"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_dis.code, 200)
        self.assertTrue(json.loads(res_dis.body.decode("utf-8"))["success"])

        # Emergency Stop API endpoint
        res_stop = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "emergency_stop"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_stop.code, 200)
        self.assertTrue(json.loads(res_stop.body.decode("utf-8"))["success"])

    def test_action_api_blocked_when_robot_disabled(self) -> None:
        """When robot is disabled, motion executions (grasp, macro_center, safe_home, ready_pose) must be blocked."""
        self.state.robot_enabled = False

        for action in ("grasp", "macro_center", "safe_home", "ready_pose"):
            res = self.fetch(
                "/api/action",
                method="POST",
                body=json.dumps({"action": action}),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(res.code, 200)
            data = json.loads(res.body.decode("utf-8"))
            self.assertFalse(data.get("success", True))
            self.assertIn("Robot disabled. Enable robot first.", data.get("error", ""))

        # Verify a warning was logged
        self.assertTrue(any("Robot disabled. Enable robot first." in log["text"] for log in self.state.logs))

        # But enable_robot is NOT blocked:
        res_en = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "enable_robot"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_en.code, 200)
        self.assertTrue(json.loads(res_en.body.decode("utf-8"))["success"])

    def test_waypoints_api_read_and_write(self) -> None:
        """GET and POST /api/waypoints can read, update, and manage waypoints."""
        res_get = self.fetch("/api/waypoints")
        self.assertEqual(res_get.code, 200)
        data = json.loads(res_get.body.decode("utf-8"))
        self.assertIn("approach", data)
        self.assertIn("retreat", data)
        self.assertGreaterEqual(len(data["retreat"]), 2)
        self.assertIn("waypoints_file", data)
        self.assertEqual(data["waypoints_file"], "waypoints.yaml")

        # Update approach waypoints with backup
        orig_content = (self.state.waypoints_cfg_path).read_text(encoding="utf-8")
        try:
            new_approach = [{"dx": 0.0, "dy": 0.0, "dz": 0.15, "duration": 2.5}]
            res_post = self.fetch(
                "/api/waypoints",
                method="POST",
                body=json.dumps({"approach": new_approach, "save_to_disk": True}),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(res_post.code, 200)
            self.assertTrue(json.loads(res_post.body.decode("utf-8"))["success"])
            self.assertEqual(len(self.state.approach_waypoints), 1)
            self.assertEqual(self.state.approach_waypoints[0]["dz"], 0.15)

            # Test save_disk action
            res_save = self.fetch(
                "/api/waypoints",
                method="POST",
                body=json.dumps({"action": "save_disk"}),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(res_save.code, 200)
            self.assertTrue(json.loads(res_save.body.decode("utf-8"))["success"])
        finally:
            (self.state.waypoints_cfg_path).write_text(orig_content, encoding="utf-8")
            self.state.reload_waypoints_from_disk()

    def test_waypoint_inline_quick_edit_html_elements(self) -> None:
        """HTML includes inline waypoint quick-edit functions, close button, and enlarged card styling."""
        res = self.fetch("/")
        self.assertEqual(res.code, 200)
        body = res.body.decode("utf-8")
        self.assertIn("openEditWaypoint", body)
        self.assertIn("closeEditWaypoint", body)
        self.assertIn("applyWaypointEdits", body)
        self.assertIn("saveEditWaypoint", body)
        self.assertIn("wp-item-editing", body)
        self.assertIn("wp-edit-close-btn", body)
        self.assertIn("Quick Modify", body)

    def test_config_api_read_and_write(self) -> None:
        """GET and POST /api/config can read and save configuration parameters safely."""
        res_get = self.fetch("/api/config")
        self.assertEqual(res_get.code, 200)
        cfg = json.loads(res_get.body.decode("utf-8"))
        self.assertIn("camera", cfg)
        self.assertIn("robot", cfg)
        self.assertIn("grasp_pipeline", cfg)
        self.assertIn("grasp", cfg["grasp_pipeline"])
        self.assertEqual(cfg["grasp_pipeline"]["grasp"].get("grasp_z_offset_m"), -0.02)
        self.assertIn("gripper", cfg["robot"])
        self.assertEqual(cfg["robot"]["gripper"]["ready_open_m"], 0.055)
        self.assertEqual(cfg["robot"]["gripper"]["close_offset_m"], 0.01)
        self.assertEqual(cfg["robot"]["gripper"]["fixed_close_m"], 0.03)

        # Test POST config with temporary backup to avoid dirtying git
        orig_content = (self.cfg_path).read_text(encoding="utf-8")
        orig_wps = (self.state.waypoints_cfg_path).read_text(encoding="utf-8")
        try:
            update_payload = {
                "grasp_pipeline": {
                    "grasp": {
                        "table_z_m": 0.175,
                        "grasp_z_offset_m": -0.035,
                    }
                },
                "camera": {"focus": 10},
                "robot": {
                    "gripper": {
                        "ready_open_m": 0.065,
                        "close_offset_m": 0.015,
                        "fixed_close_m": 0.028,
                    }
                },
            }
            res_post = self.fetch(
                "/api/config",
                method="POST",
                body=json.dumps(update_payload),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(res_post.code, 200)
            res_data = json.loads(res_post.body.decode("utf-8"))
            self.assertTrue(res_data["success"])

            # Verify on-disk file has updated values
            updated_cfg = yaml.safe_load((self.cfg_path).read_text(encoding="utf-8"))
            self.assertEqual(updated_cfg["grasp_pipeline"]["grasp"]["table_z_m"], 0.175)
            self.assertEqual(updated_cfg["grasp_pipeline"]["grasp"]["grasp_z_offset_m"], -0.035)
            self.assertEqual(updated_cfg["camera"]["focus"], 10)
            self.assertEqual(updated_cfg["robot"]["gripper"]["ready_open_m"], 0.065)
            self.assertEqual(updated_cfg["robot"]["gripper"]["close_offset_m"], 0.015)
            self.assertEqual(updated_cfg["robot"]["gripper"]["fixed_close_m"], 0.028)

            # Verify state updated
            self.assertEqual(self.state.gripper_ready_open_m, 0.065)
            self.assertEqual(self.state.gripper_close_offset_m, 0.015)
            self.assertEqual(self.state.gripper_fixed_close_m, 0.028)
        finally:
            # Restore original configuration
            (self.cfg_path).write_text(orig_content, encoding="utf-8")
            (self.state.waypoints_cfg_path).write_text(orig_wps, encoding="utf-8")
            self.state.reload_config()

    def test_gripper_control_and_test_action(self) -> None:
        """test_gripper action is guarded by robot_enabled and commands gripper appropriately."""
        # 1. When robot is disabled, test_gripper must be blocked
        self.state.robot_enabled = False
        res_dis = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "test_gripper", "width_m": 0.055}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_dis.code, 200)
        data_dis = json.loads(res_dis.body.decode("utf-8"))
        self.assertFalse(data_dis.get("success", True))
        self.assertIn("Robot disabled. Enable robot first.", data_dis.get("error", ""))

        # 2. When robot is enabled, test_gripper succeeds
        self.state.robot_enabled = True
        res_en = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "test_gripper", "width_m": 0.055}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_en.code, 200)
        data_en = json.loads(res_en.body.decode("utf-8"))
        self.assertTrue(data_en.get("success", False))

        # 3. Test worker direct execution
        worker = RobotWorker(self.state)
        worker._execute_test_gripper(0.045)
        self.assertTrue(any("Gripper width set to 4.5 cm" in log["text"] for log in self.state.logs))

    def test_status_api_error_payload(self) -> None:
        """GET /api/status returns error_count, last_error, and up to 150 log entries."""
        try:
            raise RuntimeError("API status error diagnostic test")
        except Exception as e:
            self.state.log_error(f"Test failure: {e}", exc=e, source="DiagnosticTest")

        res = self.fetch("/api/status")
        self.assertEqual(res.code, 200)
        data = json.loads(res.body.decode("utf-8"))
        self.assertIn("error_count", data)
        self.assertGreaterEqual(data["error_count"], 1)
        self.assertIsNotNone(data["last_error"])
        self.assertIn("API status error diagnostic test", data["last_error"]["text"])
        self.assertIn("traceback", data["last_error"])
        self.assertIn("RuntimeError", data["last_error"]["traceback"])
        self.assertIsInstance(data["logs"], list)

    def test_action_api_log_and_error_endpoints(self) -> None:
        """POST /api/action handles trigger_test_error, dismiss_error, and clear_logs."""
        # 1. Trigger test error
        res_trigger = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "trigger_test_error"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_trigger.code, 200)
        self.assertTrue(json.loads(res_trigger.body.decode("utf-8"))["success"])
        self.assertIsNotNone(self.state.last_error)
        self.assertIn("Diagnostic Test Error", self.state.last_error["text"])

        # 2. Dismiss error
        res_dismiss = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "dismiss_error"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_dismiss.code, 200)
        self.assertIsNone(self.state.last_error)

        # 3. Clear logs
        res_clear = self.fetch(
            "/api/action",
            method="POST",
            body=json.dumps({"action": "clear_logs"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_clear.code, 200)
        self.assertEqual(self.state.error_count, 0)
        self.assertIsNone(self.state.last_error)

    def test_auto_sync_external_config_and_ready_pose(self) -> None:
        """Modifying default.yaml on disk auto-syncs into GUI API and robot worker motions immediately."""
        orig_content = (self.cfg_path).read_text(encoding="utf-8")
        orig_wps = (self.state.waypoints_cfg_path).read_text(encoding="utf-8")
        try:
            # Modify default.yaml externally
            cfg_data = yaml.safe_load(orig_content)
            cfg_data["robot"]["ready_pose"] = {
                "x": 0.42,
                "y": 0.05,
                "z": 0.33,
                "pitch": 0.85,
                "duration": 2.5,
            }
            cfg_data["robot"]["gripper"]["ready_open_m"] = 0.062
            (self.cfg_path).write_text(yaml.dump(cfg_data), encoding="utf-8")
            # Advance mtime slightly so stat() detects modification
            new_mtime = time.time() + 2.0
            os.utime(str(self.cfg_path), (new_mtime, new_mtime))

            # 1. Fetching GET / (MainIndexHandler) triggers check_and_reload_if_modified
            res_index = self.fetch("/")
            self.assertEqual(res_index.code, 200)
            self.assertEqual(self.state.cfg["robot"]["ready_pose"]["x"], 0.42)
            self.assertEqual(self.state.cfg["robot"]["ready_pose"]["pitch"], 0.85)
            self.assertEqual(self.state.gripper_ready_open_m, 0.062)

            # 2. Fetching /api/config returns the newly loaded configuration
            res_cfg = self.fetch("/api/config")
            self.assertEqual(res_cfg.code, 200)
            cfg_resp = json.loads(res_cfg.body.decode("utf-8"))
            self.assertEqual(cfg_resp["robot"]["ready_pose"]["x"], 0.42)
            self.assertEqual(cfg_resp["robot"]["gripper"]["ready_open_m"], 0.062)

            # 3. Executing ready pose logs the exact new coordinates
            self.state.robot_enabled = True
            self.robot_worker._execute_ready_pose()
            latest_logs = [log["text"] for log in self.state.logs]
            matching_log = any("Moving arm to Ready Pose: (x=0.420, y=0.050, z=0.330, pitch=0.85, gripper=6.2cm)" in log for log in latest_logs)
            self.assertTrue(matching_log, f"Expected ready pose log not found in: {latest_logs[-5:]}")
        finally:
            (self.cfg_path).write_text(orig_content, encoding="utf-8")
            (self.state.waypoints_cfg_path).write_text(orig_wps, encoding="utf-8")
            self.state.reload_config()

    def test_teach_mode_api_get_pose(self) -> None:
        """GET /api/teach/pose returns live arm pose, joint angles, and staged waypoints."""
        response = self.fetch("/api/teach/pose")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("active", data)
        self.assertIn("joints", data)
        self.assertIn("joints_deg", data)
        self.assertIn("cartesian", data)
        self.assertEqual(len(data["joints"]), 6)
        self.assertEqual(len(data["joints_deg"]), 6)
        for k in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.assertIn(k, data["cartesian"])
        self.assertIn("gripper_width_m", data)
        self.assertIn("staged_waypoints", data)
        self.assertIn("staged_count", data)

    def test_teach_mode_lifecycle_and_safe_home(self) -> None:
        """Teach mode can be started and stopped, auto-homing the arm on deactivation."""
        # 1. Start teach mode
        res_start = self.fetch(
            "/api/teach/mode",
            method="POST",
            body=json.dumps({"action": "start"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_start.code, 200)
        self.robot_worker._execute_start_teach_mode()
        self.assertTrue(self.state.teach_mode_active)

        # 2. Stop teach mode (deactivates and runs safe home)
        res_stop = self.fetch(
            "/api/teach/mode",
            method="POST",
            body=json.dumps({"action": "stop"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_stop.code, 200)
        self.robot_worker._execute_stop_teach_mode()
        self.assertFalse(self.state.teach_mode_active)
        latest_logs = [log["text"] for log in self.state.logs]
        self.assertTrue(any("Teach Mode deactivated. Arm safely homed." in log for log in latest_logs))

    def test_teach_mode_hardware_stop_calls_gc_safe_home_and_restores_station(self) -> None:
        """When stopping teach mode on hardware, gc.safe_home() is called, arm mode is restored to posvel, and loop restarted."""
        arm_mock = unittest.mock.MagicMock()
        arm_mock.control_loop_active = False
        arm_mock.has_gripper = True
        arm_mock.arm.mode_pos_vel.return_value = True
        arm_mock.gripper.mode_mit.return_value = True

        gc_mock = unittest.mock.MagicMock()
        gc_mock._running = True

        ctrl_mock = unittest.mock.MagicMock()
        ctrl_mock._arm_control_mode = "posvel"
        ctrl_mock._running = False
        ctrl_mock._n = 6

        driver_mock = unittest.mock.MagicMock()

        self.state.dry_run = False
        self.state.arm = arm_mock
        self.state.controller = ctrl_mock
        self.state.grasp_driver = driver_mock
        self.state.teach_mode_active = True
        self.state.gravity_controller = gc_mock

        self.robot_worker._execute_stop_teach_mode()

        # 1. gc.end() and gc.safe_home() must be called
        gc_mock.end.assert_called_once()
        gc_mock.safe_home.assert_called_once_with(max_vel=0.4, send_freq=50.0, settle=0.5, timeout=15.0)
        self.assertIsNone(self.state.gravity_controller)
        self.assertFalse(self.state.teach_mode_active)

        # 2. Arm group must be restored to mode_pos_vel, gripper to mode_mit, enable_all
        arm_mock.arm.mode_pos_vel.assert_called_once()
        arm_mock.gripper.mode_mit.assert_called_once()
        arm_mock.enable_all.assert_called_once()

        # 3. Grasp driver targets primed and control loop restarted
        driver_mock._prime_arm_target.assert_called_once()
        driver_mock._prime_gripper_state.assert_called_once()
        arm_mock.start_control_loop.assert_called_once_with(driver_mock._loop_cb)
        self.assertTrue(ctrl_mock._running)

    def test_emergency_stop_aborts_active_teach_mode(self) -> None:
        """Emergency Stop while in Teach Mode safely ends gravity compensation."""
        gc_mock = unittest.mock.MagicMock()
        gc_mock._running = True
        self.state.teach_mode_active = True
        self.state.gravity_controller = gc_mock

        self.robot_worker._execute_emergency_stop()

        gc_mock.end.assert_called_once()
        self.assertFalse(self.state.teach_mode_active)
        self.assertIsNone(self.state.gravity_controller)

    def test_teach_mode_stage_and_discard_and_clear(self) -> None:
        """Staging board supports adding Cartesian or Joint poses, discarding individual items, and clearing."""
        with self.state.lock:
            self.state.staged_waypoints.clear()

        # 1. Stage in Cartesian mode
        res_cart = self.fetch(
            "/api/teach/stage",
            method="POST",
            body=json.dumps({
                "action": "stage_add",
                "mode": "cartesian",
                "duration": 2.5,
                "gripper": "open",
            }),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_cart.code, 200)
        data_cart = json.loads(res_cart.body.decode("utf-8"))
        self.assertTrue(data_cart["success"])
        self.assertEqual(data_cart["count"], 1)
        self.assertEqual(self.state.staged_waypoints[0]["mode"], "cartesian")
        self.assertIn("x", self.state.staged_waypoints[0])
        self.assertEqual(self.state.staged_waypoints[0]["duration"], 2.5)
        self.assertEqual(self.state.staged_waypoints[0]["gripper"], "open")

        # 2. Stage in Joint Angles mode
        res_joints = self.fetch(
            "/api/teach/stage",
            method="POST",
            body=json.dumps({
                "action": "stage_add",
                "mode": "joints",
                "duration": 3.0,
                "gripper": "keep",
            }),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_joints.code, 200)
        data_joints = json.loads(res_joints.body.decode("utf-8"))
        self.assertEqual(data_joints["count"], 2)
        self.assertEqual(self.state.staged_waypoints[1]["mode"], "joints")
        self.assertIn("joints", self.state.staged_waypoints[1])
        self.assertEqual(len(self.state.staged_waypoints[1]["joints"]), 6)

        # 3. Discard individual pose #0
        res_discard = self.fetch(
            "/api/teach/stage",
            method="POST",
            body=json.dumps({
                "action": "stage_discard",
                "index": 0,
            }),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_discard.code, 200)
        self.assertEqual(len(self.state.staged_waypoints), 1)
        self.assertEqual(self.state.staged_waypoints[0]["mode"], "joints")

        # 4. Clear all staged poses
        res_clear = self.fetch(
            "/api/teach/stage",
            method="POST",
            body=json.dumps({"action": "stage_clear"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_clear.code, 200)
        self.assertEqual(len(self.state.staged_waypoints), 0)

    def test_teach_mode_commit_and_waypoint_resolution(self) -> None:
        """Exporting staged points persists Cartesian and Joint waypoints and resolves them via kinematics."""
        orig_wps = (self.state.waypoints_cfg_path).read_text(encoding="utf-8")
        try:
            with self.state.lock:
                self.state.staged_waypoints = [
                    {
                        "mode": "cartesian",
                        "x": 0.360, "y": 0.050, "z": 0.220,
                        "roll": 0.0, "pitch": 0.65, "yaw": 0.0,
                        "duration": 2.2, "gripper": "keep",
                    },
                    {
                        "mode": "joints",
                        "joints": [0.0, -0.65, -0.85, 0.0, 0.45, 0.0],
                        "duration": 2.8, "gripper": "open",
                    },
                ]

            init_retreat_count = len(self.state.retreat_waypoints)

            # Commit to retreat
            res_commit = self.fetch(
                "/api/teach/commit",
                method="POST",
                body=json.dumps({"action": "commit", "target": "retreat"}),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(res_commit.code, 200)
            self.assertEqual(len(self.state.retreat_waypoints), init_retreat_count + 2)

            # Test that _resolve_waypoint_pose resolves both the Cartesian and Joint waypoints
            from scripts.web_grasp_gui import _resolve_waypoint_pose
            cart_wp = self.state.retreat_waypoints[-2]
            joint_wp = self.state.retreat_waypoints[-1]

            cx, cy, cz, cr, cp, cyaw, cdur = _resolve_waypoint_pose(cart_wp, (0.35, 0, 0.15), {"roll": 0, "pitch": 0, "yaw": 0})
            self.assertAlmostEqual(cx, 0.360, places=3)
            self.assertAlmostEqual(cdur, 2.2, places=1)

            jx, jy, jz, jr, jp, jyaw, jdur = _resolve_waypoint_pose(joint_wp, (0.35, 0, 0.15), {"roll": 0, "pitch": 0, "yaw": 0})
            self.assertAlmostEqual(jdur, 2.8, places=1)
            self.assertIsInstance(jx, float)
            self.assertIsInstance(jz, float)
        finally:
            (self.state.waypoints_cfg_path).write_text(orig_wps, encoding="utf-8")
            self.state.reload_waypoints_from_disk()

    def test_teach_mode_html_elements(self) -> None:
        """HTML must contain Tab 5 elements: side-by-side boards, slide selector, and zero emoji icons."""
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        body = response.body.decode("utf-8")
        self.assertIn("Teach &amp; Record", body)
        self.assertIn("tab-teach", body)
        self.assertIn("btn-teach-toggle", body)
        self.assertIn("teach-status-badge", body)
        self.assertIn("btn-mode-cartesian", body)
        self.assertIn("btn-mode-joints", body)
        self.assertIn("teach-hud-cartesian", body)
        self.assertIn("teach-hud-joints", body)
        self.assertIn("RECORDED WAYPOINTS BOARD", body)
        self.assertIn("btn-record-pose", body)
        self.assertIn("teach-staged-container", body)
        self.assertIn("commitTeachWaypoints", body)
        self.assertIn("discardTeachPose", body)
        self.assertIn("clearTeachStaging", body)

    def test_direct_click_to_pick_html_and_status_api(self) -> None:
        """HTML must contain direct click-to-pick feedback layer and API status returns bbox and presence."""
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        body = response.body.decode("utf-8")
        self.assertIn("video-feedback-layer", body)
        self.assertIn("handleVideoClick", body)
        self.assertIn("handleVideoMouseMove", body)
        self.assertIn("showVideoClickFeedback", body)
        self.assertIn("video-click-ripple", body)
        self.assertIn("video-click-pill", body)

        status_res = self.fetch("/api/status")
        self.assertEqual(status_res.code, 200)
        status_data = json.loads(status_res.body.decode("utf-8"))
        self.assertIn("tubes", status_data)
        if status_data["tubes"]:
            tube0 = status_data["tubes"][0]
            self.assertIn("bbox_xyxy", tube0)
            self.assertIn("center_px", tube0)


    def test_continuous_live_camera_feed_during_grasp(self) -> None:
        """When action 'grasp' is requested, state is 'executing' (not 'frozen'), keeping live stream active."""
        self.state.add_to_queue(1)
        self.state.robot_enabled = True

        res = self.fetch("/api/action", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"action": "grasp"}))
        self.assertEqual(res.code, 200)

        # Status must be 'executing', never 'frozen'
        self.assertEqual(self.state.status, "executing")
        self.assertEqual(self.state.status_detail, "Executing Picking Queue")


class TestLeftToRightNumberingAndContinuousFeed(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg_path = PROJECT_ROOT / "config" / "default.yaml"
        self.state = SystemState(self.cfg_path, dry_run=True)

    def test_simple_left_to_right_numbering(self) -> None:
        raw_tubes = _generate_synthetic_tubes(0.17)
        # Reverse order to verify sorting
        raw_tubes.reverse()
        raw_tubes.sort(key=lambda t: t["center_px"][0])
        for idx, t in enumerate(raw_tubes, start=1):
            t["id"] = idx
        self.assertEqual([t["id"] for t in raw_tubes], [1, 2, 3, 4])

    def test_snapshot_on_enqueue_behavior(self) -> None:
        raw_tubes = _generate_synthetic_tubes(0.17)
        for idx, t in enumerate(raw_tubes, start=1):
            t["id"] = idx
        self.state.detected_tubes = raw_tubes
        self.state.add_to_queue(2)
        snap_y = self.state.queue[0]["p_base"][1]
        # Mutate detected_tubes to simulate tube movement after enqueuing
        self.state.detected_tubes[1]["p_base"][1] = 0.120
        # Queued item must retain its snapshotted coordinate
        self.assertEqual(self.state.queue[0]["p_base"][1], snap_y)

    def test_robot_worker_executes_snapshotted_coordinates(self) -> None:
        raw_tubes = _generate_synthetic_tubes(0.17)
        for idx, t in enumerate(raw_tubes, start=1):
            t["id"] = idx
        self.state.detected_tubes = raw_tubes
        self.state.robot_enabled = True
        self.state.dry_run = True
        self.state.add_to_queue(1)
        worker = RobotWorker(self.state)
        worker._execute_queue()
        self.assertTrue(any("target=(0.350, -0.090, 0.150)" in l["text"] for l in self.state.logs))
        self.assertEqual(len(self.state.queue), 0)
        self.assertEqual(self.state.status, "live")


if __name__ == "__main__":
    unittest.main()
