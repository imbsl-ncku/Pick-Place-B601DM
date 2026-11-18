#!/usr/bin/env python3
"""
Web-Based Graphical User Interface for reBot Arm B601 Fixed-Path Visual Grasping.

Features:
  - Live Camera Feed with Left-to-Right Tube Indexing (#1, #2, #3...)
  - Interactive Picking Queue (select tubes in any custom order)
  - Freeze-on-Grasp Execution (locks 3D poses when Grasp is clicked)
  - Strict adherence to RETREAT_WAYPOINTS (no extra open_gripper calls)
  - Web Controls replacing keyboard shortcuts (Grasp, Macro Center, Reset/Unfreeze, Resume)
  - Dedicated Configuration Tab to edit and save config/default.yaml directly from the browser

Usage:
    python scripts/web_grasp_gui.py
    python scripts/web_grasp_gui.py --dry-run
    python scripts/web_grasp_gui.py --port 8000 --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import collections
import copy
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tornado.ioloop
import tornado.web
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from drivers.robot.grasp_driver import (
    GraspDriver,
    ensure_rebot_sdk_in_syspath,
    selected_arm_config,
)

ensure_rebot_sdk_in_syspath()

try:
    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose
    from reBotArm_control_py.controllers.gravity_compensation import GravityCompensation
    from reBotArm_control_py.kinematics import (
        joint_to_pose,
        load_robot_model,
        compute_fk,
        pad_q_for_model,
        get_end_effector_frame_id,
    )
except ImportError:
    RebotArm = None  # type: ignore
    RebotArmEndPose = None  # type: ignore
    GravityCompensation = None  # type: ignore
    joint_to_pose = None  # type: ignore
    load_robot_model = None  # type: ignore
    compute_fk = None  # type: ignore
    pad_q_for_model = None  # type: ignore
    get_end_effector_frame_id = None  # type: ignore

from utils.camera_utils import (
    compose_cam_to_base_transform,
    hand_eye_compensation_matrix,
    load_config,
    load_hand_eye,
)
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps
from utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    pose6d_to_mat4,
    rotation_matrix_to_euler_zyx,
)
from utils.yolo_utils import DetectorMemoryError, load_yolo, warmup_yolo

# Disable noisy tornado access logs
logging.getLogger("tornado.access").setLevel(logging.WARNING)


# ==============================================================================
# FIXED PATH & WAYPOINT CONFIGURATION (Matching scripts/fixed_path_grasp.py)
# ==============================================================================

LOCAL_READY_POSE: dict[str, float] = {
    "x": 0.3,
    "y": 0.0,
    "z": 0.3,
    "roll": 0.0,
    "pitch": 0.7,
    "yaw": 0.0,
    "duration": 3.0,
}

FIXED_GRASP_RPY: dict[str, float] = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}

GRIPPER_READY_OPEN_M: float = 0.055
GRIPPER_CLOSE_OFFSET_M: float = 0.01
GRIPPER_FIXED_CLOSE_M: Optional[float] = 0.03

APPROACH_WAYPOINTS: list[dict[str, Any]] = []

ENABLE_PREGRASP: bool = True
PREGRASP_OFFSET_BASE: dict[str, float] = {
    "dx": 0.00,
    "dy": 0.00,
    "dz": 0.10,
    "duration": 5.0,
}

GRASP_MOVE_DURATION: float = 2.0

# Exact retreat waypoints from fixed_path_grasp.py (no extra injected open_gripper calls)
RETREAT_WAYPOINTS: list[dict[str, Any]] = [
    {"dx": 0.00, "dy": 0.00, "dz": 0.10, "duration": 2.0},
    {"x": 0.35, "y": 0.30, "z": 0.18, "roll": -0.8, "pitch": 0.0, "yaw": 1.2, "duration": 3.0},
]


class ExecutionAborted(Exception):
    """Raised when an ongoing motion or sequence is aborted by Emergency Stop."""
    pass


def _wait_motion(
    controller: Any,
    duration: float,
    settle_time: float = 0.4,
    abort_event: Optional[threading.Event] = None,
) -> None:
    if abort_event is not None and abort_event.is_set():
        raise ExecutionAborted("Emergency stop requested.")

    if controller is None:
        if abort_event is not None:
            if abort_event.wait(timeout=duration):
                raise ExecutionAborted("Emergency stop requested during motion.")
        else:
            time.sleep(duration)
        if settle_time > 0:
            if abort_event is not None:
                if abort_event.wait(timeout=settle_time):
                    raise ExecutionAborted("Emergency stop requested during settling.")
            else:
                time.sleep(settle_time)
        return

    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        deadline = time.monotonic() + duration + 2.0
        while thread.is_alive() and time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                if hasattr(controller, "_stop_send"):
                    controller._stop_send.set()
                thread.join(timeout=0.2)
                raise ExecutionAborted("Emergency stop requested during trajectory execution.")
            thread.join(timeout=0.05)
    else:
        if abort_event is not None:
            if abort_event.wait(timeout=duration):
                raise ExecutionAborted("Emergency stop requested during motion.")
        else:
            time.sleep(duration)

    if settle_time > 0:
        if abort_event is not None:
            if abort_event.wait(timeout=settle_time):
                raise ExecutionAborted("Emergency stop requested during settling.")
        else:
            time.sleep(settle_time)


def _move_ready(
    controller: Any,
    ready_cfg: dict[str, Any],
    grasp_driver: Optional[GraspDriver] = None,
    open_at_ready: bool = True,
    dry_run: bool = False,
    ready_open_m: Optional[float] = None,
    abort_event: Optional[threading.Event] = None,
) -> None:
    if abort_event is not None and abort_event.is_set():
        raise ExecutionAborted("Emergency stop requested before ready pose motion.")
    duration = float(ready_cfg.get("duration", 3.0))
    x = float(ready_cfg.get("x", 0.35))
    y = float(ready_cfg.get("y", 0.0))
    z = float(ready_cfg.get("z", 0.28))
    roll = float(ready_cfg.get("roll", 0.0))
    pitch = float(ready_cfg.get("pitch", 0.6))
    yaw = float(ready_cfg.get("yaw", 0.0))
    if not dry_run and controller is not None:
        controller.move_to_traj(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration)
        _wait_motion(controller, duration, abort_event=abort_event)
        if abort_event is not None and abort_event.is_set():
            raise ExecutionAborted("Emergency stop requested during ready pose motion.")
        if open_at_ready and grasp_driver is not None:
            open_width = ready_open_m if ready_open_m is not None else GRIPPER_READY_OPEN_M
            grasp_driver.open_gripper(distance_m=open_width)
    else:
        if abort_event is not None:
            if abort_event.wait(timeout=0.1):
                raise ExecutionAborted("Emergency stop requested.")
        else:
            time.sleep(0.1)


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any], dry_run: bool = False) -> np.ndarray:
    if dry_run or grasp_driver is None:
        T_mock = np.eye(4, dtype=np.float64)
        T_mock[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        T_mock[2, 3] = 0.5
        return T_mock
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def _compute_camera_centering_tcp_pose(
    obj_xy: tuple[float, float],
    table_z_m: float,
    scan_z: float,
    scan_rpy: dict[str, float],
    T_hand_eye: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    rot = pose6d_to_mat4(0, 0, 0, scan_rpy["roll"], scan_rpy["pitch"], scan_rpy["yaw"])[:3, :3]
    T_comp = hand_eye_compensation_matrix(cfg)
    R_cam = rot @ T_hand_eye[:3, :3]
    dir_b = R_cam[:, 2]
    p_cam_rel = rot @ T_hand_eye[:3, 3] + T_comp[:3, 3]
    z_cam = scan_z + p_cam_rel[2]
    if abs(dir_b[2]) < 1e-6:
        return float(obj_xy[0]), float(obj_xy[1]), float(scan_z)
    t = (table_z_m - z_cam) / dir_b[2]
    tcp_x = obj_xy[0] - p_cam_rel[0] - t * dir_b[0]
    tcp_y = obj_xy[1] - p_cam_rel[1] - t * dir_b[1]
    return float(tcp_x), float(tcp_y), float(scan_z)


def _resolve_waypoint_pose(
    wp: dict[str, Any],
    grasp_xyz: tuple[float, float, float],
    default_rpy: dict[str, float],
) -> tuple[float, float, float, float, float, float, float]:
    """Resolve a waypoint dict (relative dx/dy/dz, absolute x/y/z, or joint angles) to (x, y, z, r, p, y, duration)."""
    duration = float(wp.get("duration", 2.0))

    if "joints" in wp and wp["joints"] is not None:
        q_joints = np.asarray(wp["joints"], dtype=np.float64)
        if joint_to_pose is not None:
            try:
                pos, rpy = joint_to_pose(q_joints)
                return float(pos[0]), float(pos[1]), float(pos[2]), float(rpy[0]), float(rpy[1]), float(rpy[2]), duration
            except Exception:
                pass
        return float(wp.get("x", 0.35)), float(wp.get("y", 0.0)), float(wp.get("z", 0.28)), float(default_rpy.get("roll", 0.0)), float(default_rpy.get("pitch", 0.0)), float(default_rpy.get("yaw", 0.0)), duration

    gx, gy, gz = grasp_xyz
    if "x" in wp or "y" in wp or "z" in wp:
        x = float(wp.get("x", gx))
        y = float(wp.get("y", gy))
        z = float(wp.get("z", gz))
    else:
        x = gx + float(wp.get("dx", 0.0))
        y = gy + float(wp.get("dy", 0.0))
        z = gz + float(wp.get("dz", 0.0))

    roll = float(wp.get("roll", default_rpy.get("roll", 0.0)))
    pitch = float(wp.get("pitch", default_rpy.get("pitch", 0.0)))
    yaw = float(wp.get("yaw", default_rpy.get("yaw", 0.0)))
    return x, y, z, roll, pitch, yaw, duration


def _handle_waypoint_gripper(
    grasp_driver: Optional[GraspDriver],
    gripper_cmd: Any,
    dry_run: bool = False,
    ready_open_m: Optional[float] = None,
) -> None:
    if gripper_cmd is None or dry_run or grasp_driver is None:
        return
    if isinstance(gripper_cmd, (int, float)):
        dist = float(gripper_cmd)
        grasp_driver.open_gripper(distance_m=dist)
    elif isinstance(gripper_cmd, str):
        cmd = gripper_cmd.strip().lower()
        if cmd == "open":
            open_width = ready_open_m if ready_open_m is not None else GRIPPER_READY_OPEN_M
            grasp_driver.open_gripper(distance_m=open_width)
        elif cmd == "close":
            grasp_driver.open_gripper(distance_m=0.0)


def _get_live_arm_pose(state: Any) -> dict[str, Any]:
    """Reads live joint angles and computes end-effector Cartesian pose via Forward Kinematics."""
    dry_run = getattr(state, "dry_run", True)
    arm = getattr(state, "arm", None)
    grasp_driver = getattr(state, "grasp_driver", None)

    if dry_run or arm is None:
        # Dry run simulation: generate a gentle floating motion around nominal pose
        t = time.time() - getattr(state, "sim_teach_start_time", 0.0)
        q = np.array([
            0.00 + 0.15 * np.sin(0.4 * t),
            -0.65 + 0.10 * np.cos(0.3 * t),
            -0.85 + 0.12 * np.sin(0.35 * t),
            0.00 + 0.08 * np.cos(0.5 * t),
            0.45 + 0.10 * np.sin(0.25 * t),
            0.00 + 0.20 * np.cos(0.4 * t),
        ], dtype=np.float64)
    else:
        try:
            if hasattr(arm, "arm") and hasattr(arm.arm, "get_positions"):
                q_raw = arm.arm.get_positions(request_feedback=False)
                q = np.asarray(q_raw[:6], dtype=np.float64)
            elif grasp_driver is not None and hasattr(grasp_driver, "_arm"):
                q_raw = grasp_driver._arm.get_state(request_feedback=False)[0]
                q = np.asarray(q_raw[:6], dtype=np.float64)
            else:
                q = np.zeros(6, dtype=np.float64)
        except Exception:
            q = np.zeros(6, dtype=np.float64)

    # Compute FK
    pos = np.array([0.35, 0.0, 0.28])
    rpy = np.array([0.0, 0.60, 0.0])
    if joint_to_pose is not None:
        try:
            pos, rpy = joint_to_pose(q)
        except Exception:
            pass

    # Read gripper width
    gripper_width_m = 0.055
    if not dry_run and grasp_driver is not None:
        try:
            pos_m, _, _ = grasp_driver.get_gripper_state()
            gripper_width_m = float(pos_m)
        except Exception:
            pass
    elif hasattr(state, "gripper_ready_open_m") and state.gripper_ready_open_m is not None:
        gripper_width_m = float(state.gripper_ready_open_m)

    return {
        "active": getattr(state, "teach_mode_active", False),
        "joints": [round(float(v), 4) for v in q],
        "joints_deg": [round(float(np.degrees(v)), 2) for v in q],
        "cartesian": {
            "x": round(float(pos[0]), 4),
            "y": round(float(pos[1]), 4),
            "z": round(float(pos[2]), 4),
            "roll": round(float(rpy[0]), 4),
            "pitch": round(float(rpy[1]), 4),
            "yaw": round(float(rpy[2]), 4),
            "roll_deg": round(float(np.degrees(rpy[0])), 2),
            "pitch_deg": round(float(np.degrees(rpy[1])), 2),
            "yaw_deg": round(float(np.degrees(rpy[2])), 2),
        },
        "gripper_width_m": round(float(gripper_width_m), 4),
        "gripper_width_cm": round(float(gripper_width_m * 100.0), 2),
    }


# ==============================================================================
# GLOBAL SYSTEM STATE & WORKERS
# ==============================================================================

class LogEntry(dict):
    """Structured log record that formats as a string when evaluated as str()."""

    def __str__(self) -> str:
        return self.get("text", "")


class WebGuiLoggingHandler(logging.Handler):
    """Pipes Python standard library warnings and errors into the SystemState telemetry."""

    def __init__(self, state: SystemState) -> None:
        super().__init__()
        self.state = state
        self.setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = "error" if record.levelno >= logging.ERROR else "warning"
            tb_str = None
            if record.exc_info and record.exc_info[0] is not None:
                tb_str = "".join(traceback.format_exception(*record.exc_info))
            self.state.log(
                f"[{record.name}] {record.getMessage()}",
                level=level,
                traceback_str=tb_str,
                source=record.name,
            )
        except Exception:
            self.handleError(record)


class SystemState:
    """Thread-safe state container shared across Vision, Robot, and Web API."""

    def __init__(self, cfg_path: Path, waypoints_cfg_path: Optional[Path] = None, dry_run: bool = False) -> None:
        self.cfg_path = Path(cfg_path)
        self.waypoints_cfg_path = Path(waypoints_cfg_path) if waypoints_cfg_path else (self.cfg_path.parent / "waypoints.yaml")
        self.dry_run = dry_run
        self.cfg = load_config(cfg_path)
        self.lock = threading.RLock()

        # Operational states: 'live' | 'frozen' | 'executing' | 'paused'
        self.status = "live"
        self.status_detail = "Live Camera Preview"
        self.pause_prompt = ""
        self.pause_event = threading.Event()
        self.pause_event.set()

        # Video frames
        self.latest_jpeg: Optional[bytes] = None
        self.frozen_jpeg: Optional[bytes] = None
        self.new_frame_event = threading.Event()

        # Tube tracking & queue
        self.detected_tubes: list[dict[str, Any]] = []  # sorted left-to-right
        self.queue: list[dict[str, Any]] = []          # user-picked order
        self.queue_counter = 0
        self.selected_tube_id: Optional[int] = None

        # Execution targets (locked at freeze)
        self.locked_queue: list[dict[str, Any]] = []
        self.current_execution_index = 0

        # Robot hardware handles & enablement
        self.robot_connected = False
        self.robot_enabled = False
        self.arm: Optional[Any] = None
        self.controller: Optional[Any] = None
        self.grasp_driver: Optional[GraspDriver] = None
        self.T_hand_eye: Optional[np.ndarray] = None

        # Approach and Retreat Waypoints & Motion Parameters
        self.approach_waypoints: list[dict[str, Any]] = []
        self.retreat_waypoints: list[dict[str, Any]] = []
        self.fixed_grasp_rpy: dict[str, float] = dict(FIXED_GRASP_RPY)
        self.pregrasp_config: dict[str, Any] = dict(PREGRASP_OFFSET_BASE)
        self.pregrasp_config["enabled"] = ENABLE_PREGRASP

        # Gripper configuration (in meters, matching fixed_path_grasp.py)
        self.gripper_ready_open_m: float = GRIPPER_READY_OPEN_M
        self.gripper_close_offset_m: float = GRIPPER_CLOSE_OFFSET_M
        self.gripper_fixed_close_m: Optional[float] = GRIPPER_FIXED_CLOSE_M

        # Teach & Record Mode state
        self.teach_mode_active: bool = False
        self.gravity_controller: Optional[Any] = None
        self.staged_waypoints: list[dict[str, Any]] = []
        self.sim_teach_start_time: float = time.time()

        # Activity log & Error telemetry
        self.logs: collections.deque[LogEntry] = collections.deque(maxlen=500)
        self.last_error: Optional[dict[str, Any]] = None
        self.error_count: int = 0

        # File timestamp tracking for auto-syncing external edits
        self.cfg_mtime: float = self.cfg_path.stat().st_mtime if self.cfg_path.exists() else 0.0
        self.waypoints_mtime: float = self.waypoints_cfg_path.stat().st_mtime if self.waypoints_cfg_path.exists() else 0.0

        # Load active waypoints from waypoints.yaml
        self.reload_waypoints_from_disk()
        self.log(f"System initialized. Dry run: {dry_run}")

    def log(self, msg: str, level: str = "info", traceback_str: Optional[str] = None, source: str = "") -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry_text = f"[{timestamp}] {msg}"
        entry = LogEntry({
            "time": timestamp,
            "text": entry_text,
            "message": msg,
            "level": level.lower(),
            "traceback": traceback_str,
            "source": source,
        })
        with self.lock:
            self.logs.append(entry)
            if level.lower() == "error":
                self.error_count += 1
                self.last_error = {
                    "time": timestamp,
                    "text": msg,
                    "traceback": traceback_str,
                    "source": source,
                }
        lvl_prefix = f"[{level.upper()}] " if level.lower() != "info" else ""
        print(f"[WebGUI] {lvl_prefix}{entry_text}")

    def log_error(self, msg: str, exc: Optional[BaseException] = None, source: str = "") -> None:
        tb_str = None
        if exc is not None:
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        elif sys.exc_info()[0] is not None:
            tb_str = traceback.format_exc()
        self.log(msg, level="error", traceback_str=tb_str, source=source)

    def log_warning(self, msg: str, source: str = "") -> None:
        self.log(msg, level="warning", source=source)

    def log_success(self, msg: str, source: str = "") -> None:
        self.log(msg, level="success", source=source)

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()
            self.error_count = 0
            self.last_error = None
        self.log("Telemetry logs cleared.")

    def clear_last_error(self) -> None:
        with self.lock:
            self.last_error = None

    def reload_waypoints_from_disk(self) -> None:
        """Loads approach and retreat waypoints from config/waypoints.yaml (or falls back to default.yaml/hardcoded)."""
        with self.lock:
            loaded_from_yaml = False
            if self.waypoints_cfg_path.exists():
                try:
                    with open(self.waypoints_cfg_path, "r", encoding="utf-8") as f:
                        wps_doc = yaml.safe_load(f) or {}
                    wps = wps_doc.get("waypoints", {})
                    if "approach" in wps and isinstance(wps["approach"], list):
                        self.approach_waypoints = copy.deepcopy(wps["approach"])
                    else:
                        self.approach_waypoints = copy.deepcopy(APPROACH_WAYPOINTS)

                    if "retreat" in wps and isinstance(wps["retreat"], list):
                        self.retreat_waypoints = copy.deepcopy(wps["retreat"])
                    else:
                        self.retreat_waypoints = copy.deepcopy(RETREAT_WAYPOINTS)

                    m_params = wps_doc.get("motion_parameters", {})
                    if "fixed_grasp_rpy" in m_params and isinstance(m_params["fixed_grasp_rpy"], dict):
                        self.fixed_grasp_rpy = dict(m_params["fixed_grasp_rpy"])
                    else:
                        self.fixed_grasp_rpy = dict(FIXED_GRASP_RPY)

                    if "pregrasp" in m_params and isinstance(m_params["pregrasp"], dict):
                        self.pregrasp_config = dict(m_params["pregrasp"])
                    else:
                        self.pregrasp_config = dict(PREGRASP_OFFSET_BASE)
                        self.pregrasp_config["enabled"] = ENABLE_PREGRASP

                    if "grasp" in m_params and isinstance(m_params["grasp"], dict):
                        g_p = m_params["grasp"]
                        if "ready_open_m" in g_p and g_p["ready_open_m"] is not None:
                            self.gripper_ready_open_m = float(g_p["ready_open_m"])
                        if "close_offset_m" in g_p and g_p["close_offset_m"] is not None:
                            self.gripper_close_offset_m = float(g_p["close_offset_m"])
                        if "fixed_close_m" in g_p:
                            val = g_p["fixed_close_m"]
                            self.gripper_fixed_close_m = float(val) if val is not None and str(val).strip() != "" else None

                    loaded_from_yaml = True
                    self.log(f"Loaded active waypoints from {self.waypoints_cfg_path.name}: {len(self.approach_waypoints)} approach, {len(self.retreat_waypoints)} retreat.")
                except Exception as e:
                    self.log_error(f"Failed to parse {self.waypoints_cfg_path}: {e}", exc=e, source="WaypointsLoader")

            if not loaded_from_yaml:
                cfg_wps = self.cfg.get("waypoints", {})
                self.approach_waypoints = copy.deepcopy(cfg_wps.get("approach", APPROACH_WAYPOINTS))
                self.retreat_waypoints = copy.deepcopy(cfg_wps.get("retreat", RETREAT_WAYPOINTS))
                self.fixed_grasp_rpy = dict(FIXED_GRASP_RPY)
                self.pregrasp_config = dict(PREGRASP_OFFSET_BASE)
                self.pregrasp_config["enabled"] = ENABLE_PREGRASP
                self.log(f"Using default waypoints: {len(self.approach_waypoints)} approach, {len(self.retreat_waypoints)} retreat.")

    def save_waypoints_to_disk(self) -> bool:
        """Saves current approach and retreat waypoints to config/waypoints.yaml on disk."""
        with self.lock:
            try:
                wps_doc = {}
                if self.waypoints_cfg_path.exists():
                    try:
                        with open(self.waypoints_cfg_path, "r", encoding="utf-8") as f:
                            wps_doc = yaml.safe_load(f) or {}
                    except Exception:
                        wps_doc = {}

                wps_doc.setdefault("waypoints", {})
                wps_doc["waypoints"]["approach"] = copy.deepcopy(self.approach_waypoints)
                wps_doc["waypoints"]["retreat"] = copy.deepcopy(self.retreat_waypoints)

                wps_doc.setdefault("motion_parameters", {})
                wps_doc["motion_parameters"]["fixed_grasp_rpy"] = getattr(self, "fixed_grasp_rpy", FIXED_GRASP_RPY)
                wps_doc["motion_parameters"]["pregrasp"] = getattr(self, "pregrasp_config", {
                    "enabled": ENABLE_PREGRASP,
                    "dx": PREGRASP_OFFSET_BASE.get("dx", 0.0),
                    "dy": PREGRASP_OFFSET_BASE.get("dy", 0.0),
                    "dz": PREGRASP_OFFSET_BASE.get("dz", 0.10),
                    "duration": PREGRASP_OFFSET_BASE.get("duration", 5.0),
                })
                wps_doc["motion_parameters"]["grasp"] = {
                    "move_duration": GRASP_MOVE_DURATION,
                    "ready_open_m": self.gripper_ready_open_m,
                    "close_offset_m": self.gripper_close_offset_m,
                    "fixed_close_m": self.gripper_fixed_close_m,
                }

                with open(self.waypoints_cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(wps_doc, f, default_flow_style=False, sort_keys=False)
                try:
                    self.waypoints_mtime = self.waypoints_cfg_path.stat().st_mtime
                except Exception:
                    pass
                self.log_success(f"Waypoints saved to {self.waypoints_cfg_path.name} on disk.")
                return True
            except Exception as e:
                self.log_error(f"Failed to save waypoints to {self.waypoints_cfg_path}: {e}", exc=e, source="WaypointsSaver")
                return False

    def reload_config(self) -> None:
        with self.lock:
            self.cfg = load_config(self.cfg_path)
            self.reload_waypoints_from_disk()
            rob_grp = self.cfg.get("robot", {}).get("gripper", {})
            if "ready_open_m" in rob_grp and rob_grp["ready_open_m"] is not None:
                self.gripper_ready_open_m = float(rob_grp["ready_open_m"])
            if "close_offset_m" in rob_grp and rob_grp["close_offset_m"] is not None:
                self.gripper_close_offset_m = float(rob_grp["close_offset_m"])
            if "fixed_close_m" in rob_grp:
                val = rob_grp["fixed_close_m"]
                self.gripper_fixed_close_m = float(val) if val is not None and str(val).strip() != "" else None
            try:
                if self.cfg_path.exists():
                    self.cfg_mtime = self.cfg_path.stat().st_mtime
                if self.waypoints_cfg_path.exists():
                    self.waypoints_mtime = self.waypoints_cfg_path.stat().st_mtime
            except Exception:
                pass
            self.log("Configuration and waypoints reloaded from disk.")

    def check_and_reload_if_modified(self) -> bool:
        """Checks if default.yaml or waypoints.yaml have been modified externally on disk, and reloads them."""
        with self.lock:
            modified = False
            try:
                if self.cfg_path.exists():
                    cur_cfg_mtime = self.cfg_path.stat().st_mtime
                    if cur_cfg_mtime > getattr(self, "cfg_mtime", 0.0) + 1e-4:
                        modified = True
                if self.waypoints_cfg_path.exists():
                    cur_wps_mtime = self.waypoints_cfg_path.stat().st_mtime
                    if cur_wps_mtime > getattr(self, "waypoints_mtime", 0.0) + 1e-4:
                        modified = True
            except Exception as e:
                self.log_warning(f"Error checking file timestamps: {e}", source="ConfigWatcher")

            if modified:
                self.reload_config()
                return True
            return False

    def add_to_queue(self, tube_id: int) -> bool:
        with self.lock:
            # Find the tube in current detections
            match = next((t for t in self.detected_tubes if t["id"] == tube_id), None)
            if match is None:
                return False
            self.queue_counter += 1
            item = {
                "queue_uid": self.queue_counter,
                "tube_id": tube_id,
                "label": f"Tube #{tube_id}",
                "class_name": match["class_name"],
                "p_base": match["p_base"].copy(),
                "position_cam": match["position_cam"].copy(),
                "jaw_width_m": match["jaw_width_m"],
                "center_px": match["center_px"],
            }
            self.queue.append(item)
            self.log(f"Enqueued {item['label']} (Queue Size: {len(self.queue)})")
            return True

    def remove_from_queue(self, queue_uid: int) -> None:
        with self.lock:
            self.queue = [item for item in self.queue if item["queue_uid"] != queue_uid]
            self.log(f"Removed item from queue (Remaining: {len(self.queue)})")

    def clear_queue(self) -> None:
        with self.lock:
            self.queue.clear()
            self.log("Picking queue cleared.")


# ==============================================================================
# VISION WORKER THREAD
# ==============================================================================

def _generate_synthetic_bench_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a synthetic camera bench image with a green rack and Falcon tubes for dry-run/simulation."""
    img = np.full((height, width, 3), (35, 32, 30), dtype=np.uint8)
    # Workstation grid
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (45, 42, 40), 1)
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (45, 42, 40), 1)

    # Green rack representation (green rectangle)
    cv2.rectangle(img, (140, 180), (500, 340), (40, 120, 50), -1)
    cv2.rectangle(img, (140, 180), (500, 340), (70, 180, 80), 2)
    cv2.putText(img, "GREEN TUBE RACK", (240, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

    # 4 Falcon tubes
    centers = [200, 280, 360, 440]
    for idx, cx in enumerate(centers, 1):
        # Tube cylinder body (translucent blueish)
        cv2.rectangle(img, (cx - 16, 185), (cx + 16, 295), (220, 205, 180), -1)
        cv2.rectangle(img, (cx - 16, 185), (cx + 16, 295), (160, 140, 120), 2)
        # Cap (orange)
        cv2.rectangle(img, (cx - 18, 170), (cx + 18, 190), (20, 100, 240), -1)
        # Graduation markings
        for my in range(210, 285, 15):
            cv2.line(img, (cx - 10, my), (cx - 2, my), (100, 80, 60), 1)
        # Tube label
        cv2.putText(img, f"T{idx}", (cx - 8, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 50, 20), 1)

    return img


def _generate_synthetic_tubes(table_z_m: float) -> list[dict[str, Any]]:
    """Generates synthetic detections for 4 Falcon tubes horizontally distributed."""
    centers_x = [200, 280, 360, 440]
    tubes = []
    for idx, cx in enumerate(centers_x, start=1):
        cy = 240
        w, h = 36, 120
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        # Spread along Y in robot base frame (meters)
        y_base = -0.09 + (idx - 1) * 0.06
        p_base = np.array([0.350, y_base, table_z_m], dtype=np.float64)
        tubes.append({
            "class_name": "falcon_tube",
            "conf": 0.95,
            "bbox_xyxy": [x1, y1, x2, y2],
            "center_px": (cx, cy),
            "position_cam": [0.0, 0.0, 0.45],
            "jaw_width_m": 0.03,
            "p_base": p_base,
            "rect_points": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32),
        })
    return tubes


class VisionWorker(threading.Thread):
    def __init__(self, state: SystemState) -> None:
        super().__init__(name="VisionWorker", daemon=True)
        self.state = state
        self.running = True

    def run(self) -> None:
        cfg = self.state.cfg
        cam_cfg = cfg.get("camera", {})
        self.state.log(f"Initializing camera ({cam_cfg.get('type')})...")
        last_err_time = 0.0
        cam = None
        use_synthetic_cam = False

        try:
            cam = make_camera(cfg)
            cam.open()
            cam.warm_up(5)
            self.state.log("Physical camera initialized successfully.")
        except Exception as e:
            self.state.log(f"Camera open note: {e}. Using synthetic test bench camera feed.")
            use_synthetic_cam = True

        K = None
        if cam is not None and getattr(cam, "K", None) is not None:
            K = cam.K.astype(np.float32)
        else:
            K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Load YOLO model
        yolo_cfg = cfg.get("yolo", {})
        model_name = yolo_cfg.get("model_name", "best_with_green_rack.pt")
        model = None
        yolo_opts: dict[str, Any] = {}
        try:
            self.state.log(f"Loading YOLO model: {model_name}...")
            model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)
            self.state.log("YOLO model loaded successfully.")
        except Exception as e:
            self.state.log(f"YOLO note: {e}. Using synthetic Falcon tube detector.")
            model = None

        infer_every = max(1, int(cfg.get("grasp_pipeline", {}).get("infer_every_live", 1)))
        frame_idx = 0
        last_results: list[Any] = []

        while self.running:
            try:
                if use_synthetic_cam or cam is None:
                    color_bgr = _generate_synthetic_bench_frame()
                else:
                    color_bgr, _ = cam.get_frame()
                    if color_bgr is None:
                        color_bgr = _generate_synthetic_bench_frame()

                frame_idx += 1
                with self.state.lock:
                    is_frozen = self.state.status in ("frozen", "executing", "paused")
                    T_hand_eye = self.state.T_hand_eye
                    grasp_driver = self.state.grasp_driver
                    current_cfg = self.state.cfg

                table_z_m = float(current_cfg.get("grasp_pipeline", {}).get("grasp", {}).get("table_z_m", 0.17))
                depth_quantile = float(current_cfg.get("grasp_pipeline", {}).get("grasp", {}).get("depth_quantile", 0.5))

                # Compute live camera-to-base transform
                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, current_cfg, dry_run=self.state.dry_run)

                # Detection only when live and on interval
                if not is_frozen and (frame_idx % infer_every == 0 or not self.state.detected_tubes):
                    if model is not None:
                        last_results = model.predict(
                            color_bgr,
                            verbose=False,
                            device=yolo_opts.get("device", "cpu"),
                            conf=float(yolo_opts.get("conf", 0.25)),
                            iou=float(yolo_opts.get("iou", 0.45)),
                        )

                        # Estimate candidate grasps
                        grasps = estimate_grasps(
                            last_results,
                            None,
                            K,
                            depth_quantile=depth_quantile,
                            T_cam2base=T_cam2base,
                            table_z_m=table_z_m,
                        )

                        # Filter for target items and compute 3D base coordinates
                        raw_tubes: list[dict[str, Any]] = []
                        for g in grasps:
                            if not g.is_valid:
                                continue
                            p_cam_h = np.array([g.position[0], g.position[1], g.position[2], 1.0], dtype=np.float64)
                            p_base = (T_cam2base @ p_cam_h)[:3]
                            raw_tubes.append({
                                "class_name": g.class_name,
                                "conf": g.conf,
                                "bbox_xyxy": g.bbox_xyxy,
                                "center_px": g.center_px,
                                "position_cam": g.position,
                                "jaw_width_m": g.jaw_width_m,
                                "p_base": p_base,
                                "rect_points": g.rect_points,
                            })
                    else:
                        raw_tubes = _generate_synthetic_tubes(table_z_m)

                    # SORT STRICTLY FROM LEFT TO RIGHT by horizontal pixel coordinate (center_px[0])
                    raw_tubes.sort(key=lambda t: t["center_px"][0])
                    for idx, t in enumerate(raw_tubes, start=1):
                        t["id"] = idx

                    with self.state.lock:
                        self.state.detected_tubes = raw_tubes

                # Annotate Frame
                display = color_bgr.copy()
                with self.state.lock:
                    tubes_to_draw = list(self.state.detected_tubes)
                    queued_ids = [item["tube_id"] for item in self.state.queue]
                    status_text = self.state.status.upper()
                    detail_text = self.state.status_detail

                # Clean, compact, highly legible badge sizing (proportional and balanced)
                scale_ratio = display.shape[1] / 1280.0
                font_scale = max(0.60, 0.85 * scale_ratio)
                font_thick = 2
                box_thick = max(1, int(2.0 * scale_ratio))

                # Overlay Tubes
                for t in tubes_to_draw:
                    tid = t["id"]
                    x1, y1, x2, y2 = t["bbox_xyxy"]
                    cx, cy = t["center_px"]
                    is_queued = tid in queued_ids

                    # Tube bounding box
                    box_color = (0, 255, 255) if is_queued else (0, 255, 128)
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, box_thick)

                    # Orientation polygon if available
                    if "rect_points" in t and t["rect_points"] is not None:
                        pts = np.round(t["rect_points"]).astype(np.int32)
                        cv2.polylines(display, [pts], True, (255, 200, 0), max(1, int(1.2 * scale_ratio)), cv2.LINE_AA)

                    # Number Badge: #1, #2, #3...
                    if is_queued:
                        q_pos = queued_ids.index(tid) + 1
                        badge_text = f"#{tid} [Q{q_pos}]"
                        badge_bg = (0, 140, 255)       # Amber orange
                        border_color = (255, 255, 255)
                    else:
                        badge_text = f"#{tid}"
                        badge_bg = (200, 50, 0)        # Bold blue
                        border_color = (255, 255, 255)

                    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                    pad_x = max(6, int(8 * scale_ratio))
                    pad_y = max(4, int(5 * scale_ratio))
                    badge_w = bw + 2 * pad_x
                    badge_h = bh + 2 * pad_y
                    badge_x1 = max(2, cx - badge_w // 2)
                    badge_y1 = max(2, y1 - badge_h - max(3, int(4 * scale_ratio)))
                    badge_x2 = min(display.shape[1] - 2, badge_x1 + badge_w)
                    badge_y2 = badge_y1 + badge_h

                    # Filled badge with crisp border
                    cv2.rectangle(display, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
                    cv2.rectangle(display, (badge_x1, badge_y1), (badge_x2, badge_y2), border_color, 1)

                    text_x = badge_x1 + (badge_w - bw) // 2
                    text_y = badge_y1 + bh + pad_y - 1
                    cv2.putText(display, badge_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

                    # Coordinate tooltip with dark backing
                    p_b = t["p_base"]
                    coord_text = f"X:{p_b[0]:.2f} Y:{p_b[1]:.2f} Z:{p_b[2]:.2f}"
                    coord_scale = max(0.38, 0.46 * scale_ratio)
                    (cw, ch), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, coord_scale, 1)
                    coord_y1 = min(display.shape[0] - ch - 8, y2 + max(4, int(5 * scale_ratio)))
                    cv2.rectangle(display, (x1, coord_y1 - 2), (x1 + cw + 6, coord_y1 + ch + 5), (15, 18, 24), -1)
                    cv2.rectangle(display, (x1, coord_y1 - 2), (x1 + cw + 6, coord_y1 + ch + 5), (60, 70, 90), 1)
                    cv2.putText(display, coord_text, (x1 + 3, coord_y1 + ch + 1), cv2.FONT_HERSHEY_SIMPLEX, coord_scale, (0, 255, 255) if is_queued else (220, 220, 220), 1, cv2.LINE_AA)

                # Top Status Pill Banner
                banner_color = (0, 180, 0) if not is_frozen else (0, 120, 255)
                cv2.rectangle(display, (10, 10), (280, 42), (20, 24, 33), -1)
                cv2.rectangle(display, (10, 10), (280, 42), banner_color, 1)
                cv2.circle(display, (26, 26), 6, banner_color, -1)
                cv2.putText(display, f"{status_text} | {len(tubes_to_draw)} tubes", (40, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if is_frozen:
                    cv2.putText(display, f"[ {detail_text} ]", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

                # JPEG Encode
                ok, jpeg = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    jpeg_bytes = jpeg.tobytes()
                    with self.state.lock:
                        self.state.latest_jpeg = jpeg_bytes
                        if not is_frozen:
                            self.state.frozen_jpeg = jpeg_bytes
                        self.state.new_frame_event.set()

                time.sleep(0.01)

            except Exception as e:
                now = time.time()
                if now - last_err_time > 4.0:
                    last_err_time = now
                    self.state.log_error(f"Vision capture/inference error: {e}", exc=e, source="VisionWorker")
                time.sleep(0.05)

        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass


# ==============================================================================
# ROBOT EXECUTION WORKER
# ==============================================================================

class RobotWorker(threading.Thread):
    def __init__(self, state: SystemState) -> None:
        super().__init__(name="RobotWorker", daemon=True)
        self.state = state
        self.running = True
        self.abort_event = threading.Event()
        self.command_queue: collections.deque[tuple[str, Any]] = collections.deque()
        self.command_event = threading.Event()

    def submit(self, action: str, data: Any = None) -> None:
        self.command_queue.append((action, data))
        self.command_event.set()

    def trigger_emergency_stop(self) -> None:
        """Immediately aborts any running robot execution and initiates emergency stop:
        Follows strictly: 1. STOP -> 2. RETURN HOME (open gripper) -> 3. DISABLE.
        """
        self.abort_event.set()
        self.state.pause_event.set()

        with self.state.lock:
            self.state.status = "executing"
            self.state.status_detail = "Emergency Stop: Aborting Execution"
            self.state.queue.clear()
            ctrl = self.state.controller
            driver = self.state.grasp_driver

        if ctrl is not None and hasattr(ctrl, "_stop_send"):
            try:
                ctrl._stop_send.set()
            except Exception:
                pass

        if driver is not None and hasattr(driver, "_state_lock"):
            try:
                with driver._state_lock:
                    if getattr(driver, "_grasp_result", None) is None:
                        driver._grasp_result = False
                    driver._position_reached = True
            except Exception:
                pass

        # Clear any pending commands and enqueue emergency stop immediately
        self.command_queue.clear()
        self.command_queue.appendleft(("emergency_stop", None))
        self.command_event.set()

    def _execute_emergency_stop(self) -> None:
        """
        Executes emergency stop strictly in order:
        1. STOP: Running step halted, abort flag cleared for safe homing motion.
        2. RETURN HOME: Open gripper to Ready Open Width, safely home the robot arm.
        3. DISABLE: Disable robot motors (torque released).
        """
        self.abort_event.clear()
        self.state.log_warning("[EMERGENCY STOP] Step 1/3: Execution stopped immediately.")

        with self.state.lock:
            self.state.status = "executing"
            self.state.status_detail = "Emergency Stop: Returning Home (Gripper Opening)"
            ctrl = self.state.controller
            driver = self.state.grasp_driver
            arm = self.state.arm
            is_en = self.state.robot_enabled
            dry = self.state.dry_run or arm is None
            ready_open_w = self.state.gripper_ready_open_m if self.state.gripper_ready_open_m is not None else GRIPPER_READY_OPEN_M

        if not is_en and not dry:
            self.state.log_warning("[EMERGENCY STOP] Robot motors already disabled; holding safe stopped state.")
            with self.state.lock:
                self.state.status = "live"
                self.state.status_detail = "Emergency Stop: Robot Disabled"
            return

        # Step 2: RETURN HOME with gripper state open (similar to Ready Open Width)
        self.state.log(f"[EMERGENCY STOP] Step 2/3: Opening gripper to Ready Open Width ({ready_open_w * 100:.1f} cm) and returning to Safe Home...")
        if not dry:
            if driver is not None:
                try:
                    driver.open_gripper(distance_m=ready_open_w)
                except Exception as e:
                    self.state.log_warning(f"[EMERGENCY STOP] Gripper open note: {e}")

            try:
                if ctrl is not None and getattr(ctrl, "_running", False):
                    if hasattr(ctrl, "safe_home"):
                        self.state.log("[EMERGENCY STOP] Moving arm to Safe Home pose...")
                        ctrl.safe_home()
                    else:
                        ready_cfg = self.state.cfg.get("robot", {}).get("ready_pose", LOCAL_READY_POSE)
                        _move_ready(ctrl, ready_cfg, grasp_driver=driver, open_at_ready=True, ready_open_m=ready_open_w)
                self.state.log_success("[EMERGENCY STOP] Arm returned to Safe Home.")
            except Exception as e:
                self.state.log_error(f"[EMERGENCY STOP] Safe home motion error: {e}")
        else:
            time.sleep(0.1)
            self.state.log_success(f"[EMERGENCY STOP] [Dry Run] Gripper opened to {ready_open_w * 100:.1f} cm and arm returned to Safe Home.")

        # Step 3: DISABLE robot motors
        self.state.log("[EMERGENCY STOP] Step 3/3: Disabling robot motors (torque released)...")
        with self.state.lock:
            self.state.status_detail = "Emergency Stop: Disabling Motors"

        if not dry and arm is not None:
            try:
                arm.disable_all()
                self.state.log_success("[EMERGENCY STOP] Robot hardware confirmed DISABLED (torque released).")
            except Exception as e:
                self.state.log_error(f"[EMERGENCY STOP] Failed to disable robot motors: {e}")
        else:
            self.state.log("[EMERGENCY STOP] Robot motors disabled (simulated).")

        with self.state.lock:
            self.state.robot_enabled = False
            self.state.status = "live"
            self.state.status_detail = "Emergency Stop Complete: Arm Homed & Disabled"
        self.state.log_success("[EMERGENCY STOP] Emergency stop procedure complete.")

    def run(self) -> None:
        # Initialize Robot
        robot_cfg = self.state.cfg.get("robot", {})
        ready_cfg = robot_cfg.get("ready_pose", LOCAL_READY_POSE)
        cam_cfg = self.state.cfg.get("camera", {})

        if not self.state.dry_run:
            self.state.log("Connecting to reBotArm B601-DM hardware...")
            try:
                selected = selected_arm_config(robot_cfg.get("repo_root"))
                rebotarm = RebotArm()
                controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
                grasp_driver = GraspDriver(
                    rebotarm,
                    controller,
                    gripper_config=robot_cfg.get("gripper"),
                    repo_root=robot_cfg.get("repo_root"),
                )
                grasp_driver.start()
                rebotarm.enable_all()
                with self.state.lock:
                    self.state.arm = rebotarm
                    self.state.controller = controller
                    self.state.grasp_driver = grasp_driver
                    self.state.robot_connected = True
                    self.state.robot_enabled = True
                self.state.log_success(f"Robot connected and enabled in {selected.controller_mode} mode (holding current pose).")
            except Exception as exc:
                self.state.log_error(f"Robot hardware connection failed: {exc}", exc=exc, source="RobotWorker")
                self.state.log_warning("Robot hardware unavailable; motors remain DISABLED.")
                with self.state.lock:
                    self.state.robot_connected = False
                    self.state.robot_enabled = False
        else:
            self.state.log("Running in Dry-Run Simulation mode.")
            with self.state.lock:
                self.state.robot_enabled = True

        # Load Hand-Eye Calibration
        cam_type = str(cam_cfg.get("type", "webcam")).lower()
        hand_eye_filename = self.state.cfg.get("calibration", {}).get("hand_eye_file", "hand_eye.npz")
        T_he, mode = load_hand_eye(PROJECT_ROOT, cam_type, hand_eye_filename)
        with self.state.lock:
            self.state.T_hand_eye = T_he
        if T_he is not None:
            self.state.log(f"Loaded Hand-Eye matrix from {hand_eye_filename} ({mode})")
        else:
            self.state.log("Hand-Eye calibration file not found; using identity.")

        last_hw_check = time.monotonic()
        try:
            while self.running:
                self.command_event.wait(timeout=0.1)

                # Continuous hardware health monitor (every 1.0s)
                now = time.monotonic()
                if now - last_hw_check > 1.0:
                    last_hw_check = now
                    with self.state.lock:
                        arm = self.state.arm
                        is_en = self.state.robot_enabled
                        dry = self.state.dry_run or arm is None

                    if not dry and arm is not None and is_en:
                        try:
                            unresponsive = []
                            for gname, group in getattr(arm, "groups", {}).items():
                                for jc in getattr(group, "_jcfgs", []):
                                    m = getattr(group, "_mm", {}).get(jc.name)
                                    if m is not None:
                                        st = m.get_state()
                                        if st is None or not np.all(np.isfinite([st.pos, st.vel, st.torq])):
                                            unresponsive.append(jc.name)
                                    else:
                                        unresponsive.append(jc.name)
                            if unresponsive:
                                with self.state.lock:
                                    self.state.robot_enabled = False
                                self.state.log_error(
                                    f"Hardware feedback lost for motor(s): {', '.join(unresponsive)}. Status marked DISABLED.",
                                    source="RobotMonitor",
                                )
                        except Exception as hw_err:
                            with self.state.lock:
                                self.state.robot_enabled = False
                            self.state.log_error(f"Hardware feedback monitoring error: {hw_err}", exc=hw_err, source="RobotMonitor")

                if not self.command_queue:
                    self.command_event.clear()
                    continue

                action, data = self.command_queue.popleft()

                # Block robot motions if robot is disabled
                if action in ("grasp_queue", "macro_center", "safe_home", "ready_pose", "test_gripper"):
                    with self.state.lock:
                        if not self.state.robot_enabled:
                            self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                            if self.state.status == "frozen":
                                self.state.status = "live"
                                self.state.status_detail = "Live Camera Preview"
                            continue

                try:
                    if action == "emergency_stop":
                        self._execute_emergency_stop()
                    elif action == "grasp_queue":
                        self._execute_queue()
                    elif action == "macro_center":
                        self._execute_macro_center(data)
                    elif action == "test_gripper":
                        self._execute_test_gripper(data)
                    elif action == "enable_robot":
                        self._execute_enable()
                    elif action == "disable_robot":
                        self._execute_disable()
                    elif action == "safe_home":
                        self._execute_safe_home()
                    elif action == "ready_pose":
                        self._execute_ready_pose()
                    elif action == "start_teach_mode":
                        self._execute_start_teach_mode()
                    elif action == "stop_teach_mode":
                        self._execute_stop_teach_mode()
                    elif action == "resume":
                        self.state.pause_event.set()
                    elif action == "unfreeze":
                        with self.state.lock:
                            self.state.status = "live"
                            self.state.status_detail = "Live Camera Preview"
                except ExecutionAborted as ea:
                    self.state.log_warning(f"[STOP] Action '{action}' interrupted by Emergency Stop: {ea}")
                except Exception as e:
                    self.state.log_error(f"Robot action '{action}' error: {e}", exc=e, source="RobotWorker")
                    with self.state.lock:
                        self.state.status = "live"
                        self.state.status_detail = f"Error during {action}"
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Safely shuts down the robot worker and unconditionally disables physical motors (Ctrl+C / terminal exit)."""
        self.running = False
        self.command_event.set()

        with self.state.lock:
            arm = self.state.arm
            ctrl = self.state.controller
            driver = self.state.grasp_driver
            dry = self.state.dry_run or arm is None

        if not dry and arm is not None:
            self.state.log("Terminal shutdown detected: Disabling robot motors (torque released)...", source="Shutdown")
            print("[Shutdown] Disabling robot motors (torque release)...")
            try:
                if driver is not None and hasattr(driver, "stop"):
                    driver.stop()
            except Exception as e:
                self.state.log_warning(f"Error stopping grasp driver: {e}", source="Shutdown")

            try:
                if ctrl is not None and getattr(ctrl, "_running", False):
                    if hasattr(ctrl, "rebotarm") and hasattr(ctrl.rebotarm, "stop_control_loop"):
                        ctrl.rebotarm.stop_control_loop()
                    ctrl._running = False
            except Exception as e:
                self.state.log_warning(f"Error stopping controller control loop: {e}", source="Shutdown")

            try:
                arm.disable_all()
                self.state.log_success("Robot hardware confirmed DISABLED on shutdown.", source="Shutdown")
                print("[Shutdown] Robot hardware confirmed DISABLED.")
            except Exception as e:
                self.state.log_error(f"Failed to disable robot motors on shutdown: {e}", exc=e, source="Shutdown")
                print(f"[Shutdown] Error disabling motors: {e}")

            try:
                if hasattr(arm, "disconnect"):
                    arm.disconnect()
            except Exception as e:
                pass
        else:
            self.state.log("Robot disabled on shutdown (simulated).", source="Shutdown")

        with self.state.lock:
            self.state.robot_enabled = False
            self.state.robot_connected = False

    def _execute_enable(self) -> None:
        self.state.log("Sending enable command to robot motors...")
        with self.state.lock:
            arm = self.state.arm
            dry = self.state.dry_run or arm is None

        if not dry and arm is not None:
            try:
                # 1. Send enable command to motor controllers
                arm.enable_all()
                time.sleep(0.08)

                # 2. Query hardware feedback to verify motors actually responded and are active
                missing = []
                groups = getattr(arm, "groups", {})
                for gname, group in groups.items():
                    if hasattr(group, "_request_feedback"):
                        try:
                            group._request_feedback()
                        except Exception:
                            pass
                    for jc in getattr(group, "_jcfgs", []):
                        m = getattr(group, "_mm", {}).get(jc.name)
                        if m is not None:
                            try:
                                st = m.get_state()
                                if st is None or not np.all(np.isfinite([st.pos, st.vel, st.torq])):
                                    missing.append(jc.name)
                            except Exception:
                                missing.append(jc.name)
                        else:
                            missing.append(jc.name)

                if missing:
                    raise RuntimeError(
                        f"Motor feedback verification failed. Unresponsive joint(s): {', '.join(missing)}. "
                        "Check motor power supply, CAN bus wiring, and emergency stop."
                    )

                with self.state.lock:
                    self.state.robot_enabled = True
                self.state.log_success(f"Hardware feedback verified: Robot motors ENABLED (torque active on {len(arm.joint_names)} joints).")
            except Exception as e:
                with self.state.lock:
                    self.state.robot_enabled = False
                self.state.log_error(f"Robot hardware failed to enable: {e}", exc=e, source="RobotWorker")
        else:
            self.state.log("Robot motors enabled (simulated).")
            with self.state.lock:
                self.state.robot_enabled = True

    def _execute_disable(self) -> None:
        with self.state.lock:
            is_en = self.state.robot_enabled

        if is_en:
            self.state.log("Safe homing arm before disabling motors for safety...")
            self._execute_safe_home()

        self.state.log("Sending disable command to robot motors...")
        with self.state.lock:
            arm = self.state.arm
            dry = self.state.dry_run or arm is None

        if not dry and arm is not None:
            try:
                arm.disable_all()
                self.state.log_success("Robot hardware confirmed DISABLED (torque released).")
            except Exception as e:
                self.state.log_error(f"Failed to disable robot motors: {e}", exc=e, source="RobotWorker")
        else:
            self.state.log("Robot motors disabled (simulated).")

        with self.state.lock:
            self.state.robot_enabled = False

    def _execute_queue(self) -> None:
        self.state.check_and_reload_if_modified()
        with self.state.lock:
            if not self.state.robot_enabled:
                self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                self.state.status = "live"
                return
            if not self.state.queue:
                self.state.log("Queue is empty; nothing to grasp.")
                self.state.status = "live"
                return
            # Freeze snapshot and lock targets
            self.state.status = "executing"
            self.state.status_detail = "Executing Picking Queue"
            items_to_pick = list(self.state.queue)
            approach_wps = copy.deepcopy(self.state.approach_waypoints)
            retreat_wps = copy.deepcopy(self.state.retreat_waypoints)

        ready_cfg = self.state.cfg.get("robot", {}).get("ready_pose", LOCAL_READY_POSE)
        total = len(items_to_pick)
        gp_cfg = self.state.cfg.get("grasp_pipeline", {}).get("grasp", {})
        insertion_depth = float(gp_cfg.get("insertion_depth_m", 0.0))
        grasp_z_offset = float(gp_cfg.get("grasp_z_offset_m", -0.02))
        self.state.log(f"Starting sequential execution of {total} queued tube(s)...")

        for idx, item in enumerate(items_to_pick, start=1):
            if self.abort_event.is_set():
                raise ExecutionAborted("Emergency stop requested before starting item.")
            label = item["label"]
            p_base = item["p_base"]
            jaw_width = item["jaw_width_m"]
            with self.state.lock:
                self.state.status_detail = f"Grasping {label} ({idx}/{total})"

            xg = float(p_base[0]) + insertion_depth
            yg = float(p_base[1])
            zg = float(p_base[2]) + grasp_z_offset
            self.state.log(f"--- Picking {label} ({idx}/{total}): target=({xg:.3f}, {yg:.3f}, {zg:.3f}) [z_offset={grasp_z_offset*100:+.1f}cm] ---")
            rpy_cfg = getattr(self.state, "fixed_grasp_rpy", FIXED_GRASP_RPY)
            pg_cfg = getattr(self.state, "pregrasp_config", PREGRASP_OFFSET_BASE)

            rxg, ryg, rzg = float(rpy_cfg.get("roll", 0.0)), float(rpy_cfg.get("pitch", 0.0)), float(rpy_cfg.get("yaw", 0.0))
            xp = xg + float(pg_cfg.get("dx", 0.0))
            yp = yg + float(pg_cfg.get("dy", 0.0))
            zp = zg + float(pg_cfg.get("dz", 0.10))

            grasp_xyz = (xg, yg, zg)
            ctrl = self.state.controller
            driver = self.state.grasp_driver
            dry = self.state.dry_run or ctrl is None

            try:
                # 1. Approach waypoints (if configured)
                for aw_idx, wp in enumerate(approach_wps, start=1):
                    if self.abort_event.is_set():
                        raise ExecutionAborted("Emergency stop requested during approach.")
                    if wp.get("pause", False):
                        prompt = wp.get("prompt", f"Paused at Approach WP #{aw_idx}")
                        with self.state.lock:
                            self.state.status = "paused"
                            self.state.pause_prompt = prompt
                            self.state.pause_event.clear()
                        self.state.log(f"  [PAUSED] {prompt} -> Waiting for web 'Resume' click...")
                        self.state.pause_event.wait()
                        if self.abort_event.is_set():
                            raise ExecutionAborted("Emergency stop requested during pause.")
                        with self.state.lock:
                            self.state.status = "executing"
                            self.state.status_detail = f"Grasping {label} ({idx}/{total})"
                        self.state.log("  Resumed motion.")

                    wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, rpy_cfg)
                    self.state.log(f"  Approach WP #{aw_idx}: ({wx:.3f}, {wy:.3f}, {wz:.3f})")
                    if not dry:
                        ctrl.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur)
                        _wait_motion(ctrl, wdur, abort_event=self.abort_event)
                        _handle_waypoint_gripper(driver, wp.get("gripper"), dry_run=dry, ready_open_m=self.state.gripper_ready_open_m)
                    else:
                        if self.abort_event.wait(timeout=0.2):
                            raise ExecutionAborted("Emergency stop requested.")

                # 2. Pre-Grasp Hover
                if self.abort_event.is_set():
                    raise ExecutionAborted("Emergency stop requested before pregrasp.")
                if pg_cfg.get("enabled", True):
                    pre_dur = float(pg_cfg.get("duration", 5.0))
                    self.state.log(f"  Move to Pregrasp: ({xp:.3f}, {yp:.3f}, {zp:.3f})")
                    if not dry:
                        ctrl.move_to_traj(xp, yp, zp, rxg, ryg, rzg, duration=pre_dur)
                        _wait_motion(ctrl, pre_dur, abort_event=self.abort_event)
                    else:
                        if self.abort_event.wait(timeout=0.3):
                            raise ExecutionAborted("Emergency stop requested.")

                # 3. Move to Grasp Target
                if self.abort_event.is_set():
                    raise ExecutionAborted("Emergency stop requested before grasp plunge.")
                self.state.log(f"  Move to Grasp: ({xg:.3f}, {yg:.3f}, {zg:.3f})")
                if not dry:
                    ctrl.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=GRASP_MOVE_DURATION)
                    _wait_motion(ctrl, GRASP_MOVE_DURATION, abort_event=self.abort_event)
                else:
                    if self.abort_event.wait(timeout=0.3):
                        raise ExecutionAborted("Emergency stop requested.")

                # 4. Close Gripper
                if self.abort_event.is_set():
                    raise ExecutionAborted("Emergency stop requested before gripper close.")
                fixed_close = self.state.gripper_fixed_close_m
                close_offset = self.state.gripper_close_offset_m
                target_jaw = max(0.0, float(fixed_close)) if fixed_close is not None else max(0.0, jaw_width - close_offset)
                self.state.log(f"  Closing gripper on {label} (target: {target_jaw*100:.1f}cm)...")
                if not dry and driver is not None:
                    driver.grasp(force=1.5)
                else:
                    if self.abort_event.wait(timeout=0.2):
                        raise ExecutionAborted("Emergency stop requested.")

                # 5. Strict Retreat Waypoints (as configured, no extra open_gripper)
                if self.abort_event.is_set():
                    raise ExecutionAborted("Emergency stop requested before retreat waypoints.")
                self.state.log("  Executing Retreat Waypoints...")
                for rw_idx, wp in enumerate(retreat_wps, start=1):
                    if self.abort_event.is_set():
                        raise ExecutionAborted("Emergency stop requested during retreat waypoints.")
                    if wp.get("pause", False):
                        prompt = wp.get("prompt", f"Paused at Retreat WP #{rw_idx}")
                        with self.state.lock:
                            self.state.status = "paused"
                            self.state.pause_prompt = prompt
                            self.state.pause_event.clear()
                        self.state.log(f"  [PAUSED] {prompt} -> Waiting for web 'Resume' click...")
                        self.state.pause_event.wait()
                        if self.abort_event.is_set():
                            raise ExecutionAborted("Emergency stop requested during pause.")
                        with self.state.lock:
                            self.state.status = "executing"
                            self.state.status_detail = f"Grasping {label} ({idx}/{total})"
                        self.state.log("  Resumed motion.")

                    wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, rpy_cfg)
                    self.state.log(f"  Retreat WP #{rw_idx}: ({wx:.3f}, {wy:.3f}, {wz:.3f})")
                    if not dry:
                        ctrl.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur)
                        _wait_motion(ctrl, wdur, abort_event=self.abort_event)
                        _handle_waypoint_gripper(driver, wp.get("gripper"), dry_run=dry, ready_open_m=self.state.gripper_ready_open_m)
                    else:
                        if self.abort_event.wait(timeout=0.3):
                            raise ExecutionAborted("Emergency stop requested.")

                # 6. Return to Ready Pose (Approach Waypoint 0) with open_at_ready=True
                if self.abort_event.is_set():
                    raise ExecutionAborted("Emergency stop requested before returning to ready pose.")
                self.state.log("  Returning to Ready Pose...")
                _move_ready(ctrl, ready_cfg, grasp_driver=driver, open_at_ready=True, dry_run=dry, ready_open_m=self.state.gripper_ready_open_m, abort_event=self.abort_event)

                # Remove item from state queue
                with self.state.lock:
                    self.state.queue = [q for q in self.state.queue if q["queue_uid"] != item["queue_uid"]]

            except ExecutionAborted as ea:
                self.state.log_warning(f"[STOP] Grasping aborted immediately by operator: {ea}")
                with self.state.lock:
                    self.state.queue.clear()
                raise
            except Exception as e:
                self.state.log_error(f"Grasp execution aborted on {label}: {e}", exc=e, source="RobotWorker")
                break

        with self.state.lock:
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"
        if not self.state.queue:
            self.state.log_success("All queued grasps completed successfully!")

    def _execute_macro_center(self, tube_id: Optional[int]) -> None:
        self.state.check_and_reload_if_modified()
        with self.state.lock:
            if not self.state.robot_enabled:
                self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                return
            tubes = list(self.state.detected_tubes)
            T_hand_eye = self.state.T_hand_eye
            cfg = self.state.cfg
            ctrl = self.state.controller

        target_tube = None
        if tube_id is not None:
            target_tube = next((t for t in tubes if t["id"] == tube_id), None)
        if target_tube is None and tubes:
            target_tube = tubes[0]

        if target_tube is None:
            self.state.log_warning("Macro Center: No tube found to center on.", source="RobotWorker")
            return

        obj_xy = (float(target_tube["p_base"][0]), float(target_tube["p_base"][1]))
        ready_cfg = cfg.get("robot", {}).get("ready_pose", LOCAL_READY_POSE)
        table_z_m = float(cfg.get("grasp_pipeline", {}).get("grasp", {}).get("table_z_m", 0.17))
        scan_z = float(ready_cfg.get("z", 0.28))
        scan_rpy = {"roll": float(ready_cfg.get("roll", 0.0)), "pitch": float(ready_cfg.get("pitch", 0.6)), "yaw": float(ready_cfg.get("yaw", 0.0))}

        if T_hand_eye is None:
            self.state.log_error("Macro Center: Hand-Eye matrix unavailable.", source="RobotWorker")
            return

        cx, cy, cz = _compute_camera_centering_tcp_pose(obj_xy, table_z_m, scan_z, scan_rpy, T_hand_eye, cfg)
        self.state.log(f"Macro Centering over Tube #{target_tube['id']}: TCP target=({cx:.3f}, {cy:.3f}, {cz:.3f})")

        try:
            if not self.state.dry_run and ctrl is not None:
                ctrl.move_to_traj(cx, cy, cz, scan_rpy["roll"], scan_rpy["pitch"], scan_rpy["yaw"], duration=3.0)
                _wait_motion(ctrl, 3.0, abort_event=self.abort_event)
            else:
                if self.abort_event.wait(timeout=0.5):
                    raise ExecutionAborted("Emergency stop requested during macro center.")
            self.state.log_success("Macro Centering complete.")
        except ExecutionAborted as ea:
            self.state.log_warning(f"[STOP] Macro Centering aborted by Emergency Stop: {ea}")
            raise
        except Exception as e:
            self.state.log_error(f"Macro Centering failed: {e}", exc=e, source="RobotWorker")

    def _execute_safe_home(self) -> None:
        self.state.check_and_reload_if_modified()
        with self.state.lock:
            if not self.state.robot_enabled:
                self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                return
            self.state.status = "executing"
            self.state.status_detail = "Safe Homing Arm"
        self.state.log("Initiating Safe Home...")

        ctrl = self.state.controller
        driver = self.state.grasp_driver
        dry = self.state.dry_run or ctrl is None

        if not dry:
            # 1. Release gripper first (like original fixed_path_grasp.py)
            try:
                if driver is not None:
                    self.state.log("Releasing gripper...")
                    driver.release_gripper()
            except Exception as e:
                self.state.log_warning(f"Gripper release note: {e}", source="RobotWorker")

            # 2. Smoothly home the arm using controller.safe_home()
            try:
                if ctrl is not None and getattr(ctrl, "_running", False):
                    if hasattr(ctrl, "safe_home"):
                        self.state.log("Moving arm to joint zero home pose...")
                        ctrl.safe_home()
                    else:
                        ready_cfg = self.state.cfg.get("robot", {}).get("ready_pose", LOCAL_READY_POSE)
                        _move_ready(ctrl, ready_cfg, grasp_driver=driver, open_at_ready=True, ready_open_m=self.state.gripper_ready_open_m)
                self.state.log_success("Safe Home complete. Arm is in safe home configuration.")
            except Exception as e:
                self.state.log_error(f"Safe home motion error: {e}", exc=e, source="RobotWorker")
        else:
            time.sleep(0.1)
            self.state.log_success("Safe Home complete (simulated).")

        with self.state.lock:
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"

    def _execute_ready_pose(self) -> None:
        self.state.check_and_reload_if_modified()
        with self.state.lock:
            if not self.state.robot_enabled:
                self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                return
            self.state.status = "executing"
            self.state.status_detail = "Moving to Ready Pose"
            ready_cfg = self.state.cfg.get("robot", {}).get("ready_pose", LOCAL_READY_POSE)
            rx = float(ready_cfg.get("x", 0.35))
            ry = float(ready_cfg.get("y", 0.0))
            rz = float(ready_cfg.get("z", 0.28))
            rpitch = float(ready_cfg.get("pitch", 0.6))
            gripper_w = self.state.gripper_ready_open_m

        self.state.log(f"Moving arm to Ready Pose: (x={rx:.3f}, y={ry:.3f}, z={rz:.3f}, pitch={rpitch:.2f}, gripper={gripper_w*100:.1f}cm)...")

        ctrl = self.state.controller
        driver = self.state.grasp_driver
        dry = self.state.dry_run or ctrl is None

        try:
            _move_ready(ctrl, ready_cfg, grasp_driver=driver, open_at_ready=True, dry_run=dry, ready_open_m=gripper_w, abort_event=self.abort_event)
            self.state.log_success("Arm returned to Ready Pose.")
        except ExecutionAborted as ea:
            self.state.log_warning(f"[STOP] Ready Pose motion aborted by Emergency Stop: {ea}")
            raise
        except Exception as e:
            self.state.log_error(f"Ready pose motion failed: {e}", exc=e, source="RobotWorker")

        with self.state.lock:
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"

    def _execute_test_gripper(self, target_width: Any) -> None:
        self.state.check_and_reload_if_modified()
        with self.state.lock:
            if not self.state.robot_enabled:
                self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                return
        driver = self.state.grasp_driver
        dry = self.state.dry_run or driver is None
        try:
            width = float(target_width) if target_width is not None else self.state.gripper_ready_open_m
        except (ValueError, TypeError):
            width = self.state.gripper_ready_open_m
        width = max(0.0, min(0.09, width))
        self.state.log(f"Setting gripper width to {width * 100:.1f} cm ({width:.3f} m)...")
        if not dry and driver is not None:
            try:
                driver.open_gripper(distance_m=width)
                self.state.log_success(f"Gripper width set to {width * 100:.1f} cm ({width:.3f} m).")
            except Exception as e:
                self.state.log_error(f"Failed to set gripper width: {e}", exc=e, source="RobotWorker")
        else:
            time.sleep(0.2)
            self.state.log_success(f"[Dry Run] Gripper width set to {width * 100:.1f} cm ({width:.3f} m).")

    def _execute_start_teach_mode(self) -> None:
        with self.state.lock:
            if self.state.teach_mode_active:
                return
            dry = self.state.dry_run or self.state.arm is None
            arm = self.state.arm
            ctrl = self.state.controller

        self.state.log("[TEACH] Activating Gravity Compensation / Teach Mode...")
        if not dry and arm is not None:
            try:
                if ctrl is not None and getattr(ctrl, "_running", False):
                    if hasattr(ctrl, "rebotarm") and hasattr(ctrl.rebotarm, "stop_control_loop"):
                        ctrl.rebotarm.stop_control_loop()

                if GravityCompensation is not None:
                    gc = GravityCompensation(arm)
                    gc.start()
                    with self.state.lock:
                        self.state.gravity_controller = gc
                else:
                    self.state.log_warning("[TEACH] GravityCompensation module not found. Falling back to compliant MIT mode.")
                    arm.arm.mode_mit(kp=2.0, kd=1.0)
                    arm.arm.enable()
            except Exception as e:
                self.state.log_error(f"[TEACH] Failed to activate gravity compensation: {e}", exc=e, source="TeachMode")
                return

        with self.state.lock:
            self.state.teach_mode_active = True
            self.state.sim_teach_start_time = time.time()
            self.state.status_detail = "Teach Mode Active (Arm Floating)"
        self.state.log_success("[TEACH] Gravity compensation active. You can now move the arm by hand.")

    def _execute_stop_teach_mode(self) -> None:
        with self.state.lock:
            if not self.state.teach_mode_active:
                return
            dry = self.state.dry_run or self.state.arm is None
            arm = self.state.arm
            gc = self.state.gravity_controller
            ctrl = self.state.controller

        self.state.log("[TEACH] Deactivating Teach Mode. Returning arm safely to Home...")
        if not dry and arm is not None:
            try:
                if gc is not None and getattr(gc, "_running", False):
                    gc.end()
                with self.state.lock:
                    self.state.gravity_controller = None

                if ctrl is not None:
                    if not getattr(ctrl, "_running", False):
                        ctrl.start()
                    if hasattr(ctrl, "safe_home"):
                        self.state.log("[TEACH] Executing Safe Home trajectory...")
                        ctrl.safe_home()
            except Exception as e:
                self.state.log_error(f"[TEACH] Error during teach mode deactivation/homing: {e}", exc=e, source="TeachMode")
        else:
            time.sleep(0.2)
            self.state.log_success("[TEACH] [Dry Run] Arm smoothly returned to Safe Home.")

        with self.state.lock:
            self.state.teach_mode_active = False
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"
        self.state.log_success("[TEACH] Teach Mode deactivated. Arm safely homed.")


# ==============================================================================
# TORNADO HTTP & API HANDLERS
# ==============================================================================

class VideoStreamHandler(tornado.web.RequestHandler):
    """Streams live MJPEG frames with multipart boundaries."""

    async def get(self) -> None:
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.set_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
        self.set_header("Pragma", "no-cache")
        self.set_header("Connection", "close")
        state: SystemState = self.application.settings["system_state"]

        while True:
            if not self.request.connection or self.request.connection.stream.closed():
                break

            with state.lock:
                jpeg = state.latest_jpeg if state.status == "live" else state.frozen_jpeg

            if jpeg is not None:
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                )
                try:
                    self.write(header + jpeg + b"\r\n")
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    break
                except Exception:
                    break

            await tornado.gen.sleep(0.04)


class StatusApiHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        state.check_and_reload_if_modified()
        with state.lock:
            tubes_summary = [
                {
                    "id": t["id"],
                    "class_name": t["class_name"],
                    "conf": round(float(t["conf"]), 2),
                    "x": round(float(t["p_base"][0]), 3),
                    "y": round(float(t["p_base"][1]), 3),
                    "z": round(float(t["p_base"][2]), 3),
                    "is_queued": any(q["tube_id"] == t["id"] for q in state.queue),
                }
                for t in state.detected_tubes
            ]
            queue_summary = [
                {
                    "queue_uid": q["queue_uid"],
                    "tube_id": q["tube_id"],
                    "label": q["label"],
                    "x": round(float(q["p_base"][0]), 3),
                    "y": round(float(q["p_base"][1]), 3),
                    "z": round(float(q["p_base"][2]), 3),
                }
                for q in state.queue
            ]
            data = {
                "status": state.status,
                "status_detail": state.status_detail,
                "pause_prompt": state.pause_prompt,
                "robot_connected": state.robot_connected,
                "robot_enabled": state.robot_enabled,
                "dry_run": state.dry_run,
                "tubes": tubes_summary,
                "queue": queue_summary,
                "approach_count": len(state.approach_waypoints),
                "retreat_count": len(state.retreat_waypoints),
                "waypoints_file": state.waypoints_cfg_path.name,
                "logs": list(state.logs)[-150:],
                "error_count": state.error_count,
                "last_error": state.last_error,
            }
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))


class QueueApiHandler(tornado.web.RequestHandler):
    def post(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        body = json.loads(self.request.body or "{}")
        action = body.get("action")
        if action == "add":
            tube_id = int(body.get("tube_id", 0))
            ok = state.add_to_queue(tube_id)
            self.write(json.dumps({"success": ok}))
        elif action == "remove":
            queue_uid = int(body.get("queue_uid", 0))
            state.remove_from_queue(queue_uid)
            self.write(json.dumps({"success": True}))
        elif action == "clear":
            state.clear_queue()
            self.write(json.dumps({"success": True}))
        else:
            self.set_status(400)
            self.write(json.dumps({"error": "Unknown action"}))


class ActionApiHandler(tornado.web.RequestHandler):
    def post(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        robot_worker: RobotWorker = self.application.settings["robot_worker"]
        try:
            body = json.loads(self.request.body or "{}")
        except Exception as e:
            state.log_error(f"Malformed JSON in Action request: {e}", exc=e, source="ActionAPI")
            self.set_status(400)
            self.write(json.dumps({"error": f"Malformed JSON: {e}"}))
            return

        action = body.get("action")

        # Block motion executions if robot is disabled
        if action in ("grasp", "macro_center", "safe_home", "ready_pose", "test_gripper"):
            with state.lock:
                if not state.robot_enabled:
                    state.log_warning("Robot disabled. Enable robot first.", source="RobotControl")
                    self.write(json.dumps({"success": False, "error": "Robot disabled. Enable robot first."}))
                    return

        if action == "grasp":
            with state.lock:
                if state.status == "executing":
                    self.write(json.dumps({"error": "Already executing"}))
                    return
                if not state.queue:
                    self.write(json.dumps({"error": "Queue is empty"}))
                    return
                # Freeze on grasp
                state.status = "frozen"
                state.status_detail = "Frame Frozen for Grasp Execution"
            robot_worker.submit("grasp_queue")
            self.write(json.dumps({"success": True}))

        elif action == "macro_center":
            tube_id = body.get("tube_id")
            robot_worker.submit("macro_center", int(tube_id) if tube_id is not None else None)
            self.write(json.dumps({"success": True}))

        elif action in ("emergency_stop", "stop"):
            robot_worker.trigger_emergency_stop()
            self.write(json.dumps({"success": True}))

        elif action == "enable_robot":
            robot_worker.submit("enable_robot")
            self.write(json.dumps({"success": True}))

        elif action == "disable_robot":
            robot_worker.submit("disable_robot")
            self.write(json.dumps({"success": True}))

        elif action == "safe_home":
            robot_worker.submit("safe_home")
            self.write(json.dumps({"success": True}))

        elif action == "ready_pose":
            robot_worker.submit("ready_pose")
            self.write(json.dumps({"success": True}))

        elif action == "test_gripper":
            width_m = body.get("width_m")
            robot_worker.submit("test_gripper", width_m)
            self.write(json.dumps({"success": True}))

        elif action == "unfreeze":
            with state.lock:
                state.status = "live"
                state.status_detail = "Live Camera Preview"
            self.write(json.dumps({"success": True}))

        elif action == "resume":
            robot_worker.submit("resume")
            self.write(json.dumps({"success": True}))

        elif action == "clear_logs":
            state.clear_logs()
            self.write(json.dumps({"success": True}))

        elif action == "dismiss_error":
            state.clear_last_error()
            self.write(json.dumps({"success": True}))

        elif action == "trigger_test_error":
            try:
                raise RuntimeError("Simulated Hardware/Communication Exception: timeout waiting for motor reply on Axis 2 (diagnostic test)")
            except Exception as e:
                state.log_error(f"Diagnostic Test Error: {e}", exc=e, source="TestDiagnostics")
            self.write(json.dumps({"success": True}))

        else:
            self.set_status(400)
            self.write(json.dumps({"error": "Unknown action"}))


class WaypointsApiHandler(tornado.web.RequestHandler):
    """GET and POST endpoints to view, add, remove, and persist approaching and retreating waypoints."""

    def get(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        state.check_and_reload_if_modified()
        with state.lock:
            data = {
                "approach": state.approach_waypoints,
                "retreat": state.retreat_waypoints,
                "waypoints_file": state.waypoints_cfg_path.name,
                "waypoints_file_path": str(state.waypoints_cfg_path),
            }
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))

    def post(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        try:
            body = json.loads(self.request.body or "{}")
            action = body.get("action", "update")
            with state.lock:
                if action == "save_disk":
                    ok = state.save_waypoints_to_disk()
                    self.write(json.dumps({"success": ok, "file": state.waypoints_cfg_path.name}))
                    return

                if action == "reload":
                    state.reload_waypoints_from_disk()
                    self.write(json.dumps({
                        "success": True,
                        "approach": state.approach_waypoints,
                        "retreat": state.retreat_waypoints,
                        "file": state.waypoints_cfg_path.name,
                    }))
                    return

                if "approach" in body and isinstance(body["approach"], list):
                    state.approach_waypoints = copy.deepcopy(body["approach"])
                if "retreat" in body and isinstance(body["retreat"], list):
                    state.retreat_waypoints = copy.deepcopy(body["retreat"])

                save_disk = body.get("save_to_disk", True)
                if save_disk:
                    state.save_waypoints_to_disk()
                else:
                    state.log("Waypoints updated in runtime memory.")

            self.write(json.dumps({"success": True, "message": "Waypoints updated successfully."}))
        except Exception as e:
            state.log_error(f"Waypoints API error: {e}", exc=e, source="WaypointsAPI")
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


class ConfigApiHandler(tornado.web.RequestHandler):
    """GET and POST endpoints to view and modify config/default.yaml."""

    def get(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        state.check_and_reload_if_modified()
        with state.lock:
            try:
                with open(state.cfg_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
            except Exception:
                content = copy.deepcopy(state.cfg)
            rob = content.setdefault("robot", {})
            grp = rob.setdefault("gripper", {})
            grp["ready_open_m"] = state.gripper_ready_open_m
            grp["close_offset_m"] = state.gripper_close_offset_m
            grp["fixed_close_m"] = state.gripper_fixed_close_m
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(content))

    def post(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        try:
            updates = json.loads(self.request.body or "{}")
            with state.lock:
                with open(state.cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                # Safely merge nested updates
                if "camera" in updates:
                    cam_up = updates["camera"]
                    if "id" in cam_up:
                        val = cam_up["id"]
                        cam_up["id"] = int(val) if isinstance(val, str) and val.isdigit() else val
                    data.setdefault("camera", {}).update(cam_up)
                if "grasp_pipeline" in updates:
                    gp = data.setdefault("grasp_pipeline", {})
                    if "grasp" in updates["grasp_pipeline"]:
                        gp.setdefault("grasp", {}).update(updates["grasp_pipeline"]["grasp"])
                if "calibration" in updates:
                    cal = data.setdefault("calibration", {})
                    if "hand_eye_file" in updates["calibration"]:
                        cal["hand_eye_file"] = updates["calibration"]["hand_eye_file"]
                    if "hand_eye_compensation_m" in updates["calibration"]:
                        cal.setdefault("hand_eye_compensation_m", {}).update(updates["calibration"]["hand_eye_compensation_m"])
                if "robot" in updates:
                    rob = data.setdefault("robot", {})
                    if "ready_pose" in updates["robot"]:
                        rob.setdefault("ready_pose", {}).update(updates["robot"]["ready_pose"])
                    if "gripper" in updates["robot"]:
                        grp = rob.setdefault("gripper", {})
                        grp_up = updates["robot"]["gripper"]
                        grp.update(grp_up)
                        if "ready_open_m" in grp_up and grp_up["ready_open_m"] is not None:
                            state.gripper_ready_open_m = float(grp_up["ready_open_m"])
                        if "close_offset_m" in grp_up and grp_up["close_offset_m"] is not None:
                            state.gripper_close_offset_m = float(grp_up["close_offset_m"])
                        if "fixed_close_m" in grp_up:
                            val = grp_up["fixed_close_m"]
                            state.gripper_fixed_close_m = float(val) if val is not None and str(val).strip() != "" else None

                # Write to disk
                with open(state.cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

                try:
                    state.cfg_mtime = state.cfg_path.stat().st_mtime
                except Exception:
                    pass

                state.reload_config()
                state.save_waypoints_to_disk()
                self.write(json.dumps({"success": True, "message": "Configuration saved successfully."}))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


class TeachApiHandler(tornado.web.RequestHandler):
    """Endpoints for live arm pose telemetry, gravity compensation mode toggle, and waypoint staging/committing."""

    def get(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        pose_info = _get_live_arm_pose(state)
        with state.lock:
            pose_info["staged_waypoints"] = copy.deepcopy(state.staged_waypoints)
            pose_info["staged_count"] = len(state.staged_waypoints)
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(pose_info))

    def post(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        robot_worker: RobotWorker = self.application.settings["robot_worker"]
        try:
            body = json.loads(self.request.body or "{}")
            action = body.get("action", "")

            if action == "start":
                robot_worker.submit("start_teach_mode")
                self.write(json.dumps({"success": True, "message": "Starting Teach Mode (Gravity Compensation)..."}))
                return

            if action == "stop":
                robot_worker.submit("stop_teach_mode")
                self.write(json.dumps({"success": True, "message": "Stopping Teach Mode and homing safely..."}))
                return

            if action == "stage_add":
                mode = body.get("mode", "cartesian")
                duration = float(body.get("duration", 2.0))
                gripper = body.get("gripper", "keep")
                live = _get_live_arm_pose(state)

                with state.lock:
                    if mode == "joints":
                        staged_item = {
                            "mode": "joints",
                            "joints": [float(v) for v in live["joints"]],
                            "joints_deg": [float(v) for v in live["joints_deg"]],
                            "duration": duration,
                            "gripper": gripper,
                            "cartesian_ref": dict(live["cartesian"]),
                        }
                    else:
                        staged_item = {
                            "mode": "cartesian",
                            "x": float(live["cartesian"]["x"]),
                            "y": float(live["cartesian"]["y"]),
                            "z": float(live["cartesian"]["z"]),
                            "roll": float(live["cartesian"]["roll"]),
                            "pitch": float(live["cartesian"]["pitch"]),
                            "yaw": float(live["cartesian"]["yaw"]),
                            "duration": duration,
                            "gripper": gripper,
                            "joints_ref": [float(v) for v in live["joints_deg"]],
                        }
                    state.staged_waypoints.append(staged_item)
                    count = len(state.staged_waypoints)
                state.log(f"[TEACH] Recorded waypoint #{count} in [{mode.upper()}] mode.")
                self.write(json.dumps({
                    "success": True,
                    "staged_waypoints": state.staged_waypoints,
                    "count": count,
                }))
                return

            if action == "stage_discard":
                index = int(body.get("index", -1))
                with state.lock:
                    if 0 <= index < len(state.staged_waypoints):
                        removed = state.staged_waypoints.pop(index)
                        state.log(f"[TEACH] Discarded staged waypoint #{index + 1} ({removed.get('mode', 'cartesian')}).")
                    count = len(state.staged_waypoints)
                self.write(json.dumps({
                    "success": True,
                    "staged_waypoints": state.staged_waypoints,
                    "count": count,
                }))
                return

            if action == "stage_clear":
                with state.lock:
                    state.staged_waypoints.clear()
                state.log("[TEACH] Cleared all staged waypoints.")
                self.write(json.dumps({
                    "success": True,
                    "staged_waypoints": [],
                    "count": 0,
                }))
                return

            if action == "commit":
                target = body.get("target", "retreat")
                with state.lock:
                    if not state.staged_waypoints:
                        self.set_status(400)
                        self.write(json.dumps({"success": False, "error": "No staged waypoints to commit."}))
                        return

                    items_to_commit = []
                    for sw in state.staged_waypoints:
                        wp_entry = {}
                        if sw.get("mode") == "joints":
                            wp_entry["joints"] = [round(float(v), 4) for v in sw["joints"]]
                        else:
                            wp_entry["x"] = round(float(sw["x"]), 4)
                            wp_entry["y"] = round(float(sw["y"]), 4)
                            wp_entry["z"] = round(float(sw["z"]), 4)
                            wp_entry["roll"] = round(float(sw.get("roll", 0.0)), 4)
                            wp_entry["pitch"] = round(float(sw.get("pitch", 0.0)), 4)
                            wp_entry["yaw"] = round(float(sw.get("yaw", 0.0)), 4)
                        wp_entry["duration"] = round(float(sw.get("duration", 2.0)), 2)
                        if sw.get("gripper") and sw["gripper"] != "keep":
                            wp_entry["gripper"] = sw["gripper"]
                        items_to_commit.append(wp_entry)

                    if target == "approach":
                        state.approach_waypoints.extend(items_to_commit)
                        state.save_waypoints_to_disk()
                        state.log_success(f"[TEACH] Added {len(items_to_commit)} waypoint(s) to Approach sequence.")
                    elif target == "retreat":
                        state.retreat_waypoints.extend(items_to_commit)
                        state.save_waypoints_to_disk()
                        state.log_success(f"[TEACH] Added {len(items_to_commit)} waypoint(s) to Retreat sequence.")
                    elif target == "ready_pose":
                        last_item = state.staged_waypoints[-1]
                        if last_item.get("mode") == "joints":
                            pos = np.array([0.35, 0.0, 0.28])
                            rpy = np.array([0.0, 0.60, 0.0])
                            if joint_to_pose is not None:
                                try:
                                    pos, rpy = joint_to_pose(np.array(last_item["joints"]))
                                except Exception:
                                    pass
                            rx, ry, rz = pos
                            rr, rp, ryaw = rpy
                        else:
                            rx, ry, rz = last_item["x"], last_item["y"], last_item["z"]
                            rr, rp, ryaw = last_item["roll"], last_item["pitch"], last_item["yaw"]

                        if "robot" not in state.cfg:
                            state.cfg["robot"] = {}
                        state.cfg["robot"]["ready_pose"] = {
                            "x": round(float(rx), 4),
                            "y": round(float(ry), 4),
                            "z": round(float(rz), 4),
                            "roll": round(float(rr), 4),
                            "pitch": round(float(rp), 4),
                            "yaw": round(float(ryaw), 4),
                        }
                        with open(state.cfg_path, "w", encoding="utf-8") as f:
                            yaml.dump(state.cfg, f, default_flow_style=False, sort_keys=False)
                        state.log_success(f"[TEACH] Ready Pose updated in {state.cfg_path.name}: ({rx:.3f}, {ry:.3f}, {rz:.3f})")

                self.write(json.dumps({"success": True, "message": f"Successfully exported to {target}."}))
                return

            self.set_status(400)
            self.write(json.dumps({"error": f"Unknown action: {action}"}))
        except Exception as e:
            state.log_error(f"Teach API error: {e}", exc=e, source="TeachAPI")
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


# ==============================================================================
# HTML FRONTEND DASHBOARD
# ==============================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>reBot Arm B601 - Pick &amp; Place Web Station</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #131a29;
      --card-bg-elevated: #1a2234;
      --card-border: rgba(255, 255, 255, 0.08);
      --card-border-subtle: rgba(255, 255, 255, 0.05);
      --surface-inset: #090d16;
      --surface-inset-hover: #0f1524;
      --primary: #0a84ff;
      --primary-hover: #0077ed;
      --primary-active: #0062cc;
      --success: #30d158;
      --success-hover: #28b84c;
      --warning: #ff9f0a;
      --warning-hover: #e08b08;
      --danger: #ff453a;
      --danger-hover: #e0382e;
      --cyan: #38bdf8;
      --cyan-muted: #7dd3fc;
      --text-primary: #f5f5f7;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --text-faint: #64748b;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: var(--font-sans);
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    body {
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Top Navigation Bar */
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--card-border);
      padding: 0 24px;
      height: 56px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }
    .brand {
      display: flex;
      align-items: center;
      font-weight: 600;
      font-size: 0.95rem;
      letter-spacing: -0.01em;
      color: var(--text-primary);
      flex-shrink: 0;
    }
    .nav-tabs {
      display: inline-flex;
      background: var(--bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 3px;
      gap: 2px;
      align-items: center;
    }
    .tab-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      font-weight: 500;
      font-size: 0.82rem;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .tab-btn.active {
      background: var(--card-bg-elevated);
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.1);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
      font-weight: 600;
    }
    .tab-btn:hover:not(.active) {
      color: var(--text-primary);
    }
    .header-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-shrink: 0;
    }
    .btn-guide-pill {
      background: rgba(10, 132, 255, 0.08);
      border: 1px solid rgba(10, 132, 255, 0.25);
      color: #60a5fa;
      padding: 4px 12px;
      font-size: 0.76rem;
      font-weight: 600;
      border-radius: 999px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn-guide-pill:hover {
      background: rgba(10, 132, 255, 0.16);
      border-color: rgba(10, 132, 255, 0.4);
    }

    /* Status Pills */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .status-live { background: rgba(48, 209, 88, 0.12); color: var(--success); border: 1px solid rgba(48, 209, 88, 0.28); }
    .status-frozen { background: rgba(10, 132, 255, 0.12); color: #60a5fa; border: 1px solid rgba(10, 132, 255, 0.28); }
    .status-executing { background: rgba(255, 159, 10, 0.12); color: var(--warning); border: 1px solid rgba(255, 159, 10, 0.28); }
    .status-paused { background: rgba(255, 69, 58, 0.15); color: var(--danger); border: 1px solid var(--danger); animation: pulse 1.5s infinite; }
    .status-waypoints {
      background: rgba(99, 102, 241, 0.1);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.22);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .status-waypoints:hover {
      background: rgba(99, 102, 241, 0.18);
      border-color: rgba(99, 102, 241, 0.4);
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* Robot Hardware LED Indicator */
    .robot-led-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      background: var(--bg);
      border: 1px solid var(--card-border);
      transition: all 0.25s ease;
      white-space: nowrap;
    }
    .led-bulb {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      transition: all 0.25s ease;
      flex-shrink: 0;
    }
    .led-bulb.led-on {
      background-color: var(--success);
      box-shadow: 0 0 6px var(--success), 0 0 12px rgba(48, 209, 88, 0.6);
      border: 1px solid #34d399;
    }
    .led-bulb.led-off {
      background-color: #475569;
      box-shadow: none;
      border: 1px solid #334155;
    }
    .led-bulb.led-pending {
      background-color: var(--warning);
      box-shadow: 0 0 6px var(--warning);
      border: 1px solid #fbbf24;
      animation: led-pulse 0.8s infinite alternate;
    }
    @keyframes led-pulse {
      0% { opacity: 0.35; }
      100% { opacity: 1; }
    }

    /* Main Container & Tabs */
    main {
      flex: 1;
      padding: 24px;
      max-width: 1520px;
      margin: 0 auto;
      width: 100%;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 18px 20px;
      margin-bottom: 20px;
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .card-title {
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Station Layout */
    .station-grid {
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 1100px) {
      .station-grid { grid-template-columns: 1fr; }
    }

    /* Video Frame */
    .video-container {
      position: relative;
      background: #000;
      border-radius: var(--radius-md);
      overflow: hidden;
      aspect-ratio: 16 / 9;
      display: flex;
      justify-content: center;
      align-items: center;
      border: 1px solid var(--card-border);
    }
    .video-container img { width: 100%; height: 100%; object-fit: contain; }

    /* Tubes Button Bar */
    .tube-buttons-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      min-height: 46px;
      align-items: center;
    }
    .tube-btn {
      background: var(--card-bg-elevated);
      color: var(--text-primary);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 8px 18px;
      min-width: 58px;
      min-height: 40px;
      border-radius: var(--radius-sm);
      font-weight: 600;
      cursor: pointer;
      font-size: 0.92rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
      transition: all 0.12s ease;
      user-select: none;
      -webkit-user-select: none;
      touch-action: manipulation;
    }
    .tube-btn:hover {
      background: #232e45;
      border-color: rgba(10, 132, 255, 0.4);
      color: #60a5fa;
      transform: translateY(-1px);
    }
    .tube-btn:active {
      transform: scale(0.94);
    }
    .tube-btn.queued {
      background: rgba(48, 209, 88, 0.14);
      color: var(--success);
      border-color: rgba(48, 209, 88, 0.35);
      box-shadow: 0 0 10px rgba(48, 209, 88, 0.15);
    }
    .tube-btn.queued:hover {
      background: rgba(48, 209, 88, 0.22);
      border-color: rgba(48, 209, 88, 0.5);
    }

    /* Queue List */
    .queue-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 190px;
      overflow-y: auto;
      margin-bottom: 14px;
    }
    .queue-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--surface-inset);
      border: 1px solid var(--card-border-subtle);
      padding: 7px 12px;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
    }
    .queue-item-info { font-weight: 500; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    .queue-item-coord { color: var(--text-muted); font-size: 0.78rem; margin-left: 8px; }
    .queue-item-remove {
      background: transparent;
      color: var(--text-muted);
      border: none;
      cursor: pointer;
      font-size: 0.95rem;
      padding: 2px 6px;
      border-radius: 4px;
      transition: all 0.15s ease;
    }
    .queue-item-remove:hover {
      color: var(--danger);
      background: rgba(255, 69, 58, 0.1);
    }
    .empty-queue-msg { color: var(--text-faint); font-size: 0.82rem; font-style: italic; }

    /* Actions Panel */
    .actions-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 16px;
    }
    .btn-main {
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--card-border);
      background: var(--card-bg-elevated);
      color: var(--text);
      font-weight: 500;
      font-size: 0.88rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    .btn-main:hover {
      background: #232e45;
      border-color: rgba(255, 255, 255, 0.16);
      color: #ffffff;
    }
    .btn-main:active {
      transform: scale(0.98);
    }
    .btn-grasp {
      background: var(--primary);
      border-color: rgba(255, 255, 255, 0.15);
      color: #ffffff;
      grid-column: span 2;
      font-size: 0.96rem;
      font-weight: 600;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(10, 132, 255, 0.35);
    }
    .btn-grasp:hover {
      background: var(--primary-hover);
      border-color: rgba(255, 255, 255, 0.25);
      box-shadow: 0 4px 14px rgba(10, 132, 255, 0.45);
    }
    .btn-grasp:disabled {
      opacity: 0.35;
      cursor: not-allowed;
      background: #1e293b;
      border-color: var(--card-border);
      box-shadow: none;
    }
    .btn-stop {
      background: var(--danger);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #ffffff;
      grid-column: span 2;
      font-size: 0.92rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      padding: 11px;
      box-shadow: 0 2px 8px rgba(255, 69, 58, 0.3);
    }
    .btn-stop:hover {
      background: var(--danger-hover);
      border-color: rgba(255, 255, 255, 0.35);
      box-shadow: 0 0 16px rgba(255, 69, 58, 0.45);
    }
    .btn-enable {
      background: var(--card-bg-elevated);
      color: var(--text);
      border-color: var(--card-border);
    }
    .btn-enable:hover {
      background: #232e45;
      border-color: rgba(48, 209, 88, 0.4);
      color: var(--success);
    }
    .btn-disable {
      background: var(--card-bg-elevated);
      color: var(--text);
      border-color: var(--card-border);
    }
    .btn-disable:hover {
      background: #232e45;
      border-color: rgba(255, 69, 58, 0.4);
      color: var(--danger);
    }
    .btn-center, .btn-home, .btn-ready, .btn-reset {
      background: var(--card-bg-elevated);
      color: var(--text);
      border-color: var(--card-border);
    }
    .btn-center:hover, .btn-home:hover, .btn-ready:hover, .btn-reset:hover {
      background: #232e45;
      border-color: rgba(255, 255, 255, 0.16);
      color: #ffffff;
    }
    .btn-resume {
      background: var(--warning);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #000000;
      font-weight: 700;
      grid-column: span 2;
      display: none;
    }
    .btn-resume:hover { background: var(--warning-hover); }
    .btn-resume.active { display: flex; animation: pulse 1s infinite; }

    /* Waypoint Editor Tab Styles */
    .wp-editor-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 960px) {
      .wp-editor-grid { grid-template-columns: 1fr; }
    }
    .wps-table-container {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 240px;
      overflow-y: auto;
      margin-bottom: 16px;
      background: var(--surface-inset);
      border: 1px solid var(--card-border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px;
    }
    .wp-item-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg-elevated);
      padding: 8px 12px;
      border-radius: var(--radius-sm);
      font-size: 0.82rem;
      border: 1px solid var(--card-border-subtle);
      transition: all 0.15s ease;
    }
    .wp-item-row:hover {
      border-color: rgba(255, 255, 255, 0.1);
      background: #202a3f;
    }
    .wp-item-desc { display: flex; flex-direction: column; gap: 3px; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    .wp-badge-mode {
      font-size: 0.68rem;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
      text-transform: uppercase;
      display: inline-block;
      width: fit-content;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .wp-badge-rel { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.28); }
    .wp-badge-abs { background: rgba(48, 209, 88, 0.12); color: #30d158; border: 1px solid rgba(48, 209, 88, 0.28); }
    .wp-form-box {
      background: var(--surface-inset);
      border: 1px solid var(--card-border-subtle);
      border-radius: var(--radius-sm);
      padding: 14px;
    }
    .wp-form-title {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .btn-sm {
      padding: 5px 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--card-border);
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn-sm:hover {
      border-color: rgba(255, 255, 255, 0.18);
    }

    /* Console & Debugging Log */
    .console-box {
      background: var(--surface-inset);
      border: 1px solid var(--card-border-subtle);
      border-radius: var(--radius-sm);
      padding: 12px 14px;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-variant-numeric: tabular-nums;
      line-height: 1.55;
      color: var(--text-muted);
      height: 320px;
      min-height: 220px;
      max-height: 600px;
      overflow-y: auto;
      resize: vertical;
      box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.4);
    }
    .console-box::-webkit-scrollbar { width: 6px; }
    .console-box::-webkit-scrollbar-thumb {
      background: #1e293b;
      border-radius: 3px;
    }
    .console-box::-webkit-scrollbar-thumb:hover { background: #334155; }

    .log-line {
      display: flex;
      flex-direction: column;
      padding: 3px 6px;
      margin-bottom: 2px;
      border-radius: 4px;
      border-left: 3px solid transparent;
      word-break: break-word;
      transition: background 0.12s;
    }
    .log-line:hover { background: rgba(255, 255, 255, 0.03); }
    .log-row-main { display: flex; gap: 8px; align-items: baseline; }
    .log-time {
      color: var(--text-faint);
      font-size: 0.76rem;
      flex-shrink: 0;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
    }
    .log-badge {
      font-size: 0.66rem;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 3px;
      text-transform: uppercase;
      flex-shrink: 0;
      font-family: var(--font-mono);
    }
    .log-msg { flex: 1; }

    .log-info { border-left-color: var(--cyan); color: var(--text); }
    .log-info .log-badge { background: rgba(56, 189, 248, 0.12); color: var(--cyan); }
    .log-success { border-left-color: var(--success); background: rgba(48, 209, 88, 0.04); color: #86efac; }
    .log-success .log-badge { background: rgba(48, 209, 88, 0.18); color: var(--success); }
    .log-warning { border-left-color: var(--warning); background: rgba(255, 159, 10, 0.05); color: #fde68a; }
    .log-warning .log-badge { background: rgba(255, 159, 10, 0.18); color: var(--warning); }
    .log-error { border-left-color: var(--danger); background: rgba(255, 69, 58, 0.08); color: #fca5a5; font-weight: 500; }
    .log-error .log-badge { background: rgba(255, 69, 58, 0.22); color: var(--danger); }

    .log-trace-details { margin-top: 4px; margin-left: 12px; font-size: 0.78rem; }
    .log-trace-details summary { color: var(--danger); cursor: pointer; font-weight: 500; text-decoration: underline; user-select: none; }
    .log-trace-pre {
      background: #020408;
      border: 1px solid rgba(255, 69, 58, 0.3);
      border-radius: 4px;
      padding: 8px 10px;
      margin-top: 4px;
      color: #fca5a5;
      font-size: 0.76rem;
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 200px;
    }

    /* Active Error Banner */
    .active-error-banner {
      background: rgba(255, 69, 58, 0.08);
      border: 1px solid rgba(255, 69, 58, 0.22);
      border-left: 3px solid var(--danger);
      border-radius: var(--radius-sm);
      padding: 10px 14px;
      margin-bottom: 12px;
    }
    .error-banner-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
    .error-banner-title { font-weight: 700; font-size: 0.8rem; color: var(--danger); letter-spacing: 0.05em; }
    .error-banner-time { font-size: 0.72rem; color: var(--text-faint); margin-left: auto; font-family: var(--font-mono); }
    .error-banner-dismiss {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 1rem;
      cursor: pointer;
      padding: 0 4px;
      line-height: 1;
    }
    .error-banner-dismiss:hover { color: #fff; }
    .error-banner-msg {
      color: #fecaca;
      font-size: 0.84rem;
      font-weight: 500;
      font-family: var(--font-mono);
      margin-left: 4px;
      word-break: break-word;
    }
    .error-trace-details { margin-top: 6px; margin-left: 4px; }
    .error-trace-details summary { color: var(--danger); font-size: 0.76rem; font-weight: 500; cursor: pointer; user-select: none; }
    .error-trace-pre {
      background: #020408;
      border: 1px solid rgba(255, 69, 58, 0.25);
      border-radius: 4px;
      padding: 8px 10px;
      margin-top: 4px;
      color: #fca5a5;
      font-size: 0.76rem;
      font-family: var(--font-mono);
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 220px;
    }

    .log-pill {
      font-size: 0.7rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 12px;
      display: inline-block;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .log-pill-clean { background: rgba(48, 209, 88, 0.12); color: var(--success); border: 1px solid rgba(48, 209, 88, 0.25); }
    .log-pill-alert { background: rgba(255, 69, 58, 0.15); color: var(--danger); border: 1px solid var(--danger); animation: pulse 1.5s infinite; }

    .log-filter-group {
      display: flex;
      background: var(--surface-inset);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .btn-filter {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.74rem;
      font-weight: 500;
      padding: 4px 10px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-filter.active { background: var(--card-bg-elevated); color: #fff; }
    .btn-filter:hover:not(.active) { color: var(--text); }

    /* Configuration Tab */
    .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
    .form-group { margin-bottom: 12px; }
    .form-group label {
      display: block;
      font-size: 0.76rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 5px;
    }
    .form-group input, .form-group select {
      width: 100%;
      background: var(--surface-inset);
      border: 1px solid var(--card-border);
      color: var(--text-primary);
      padding: 8px 10px;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      transition: all 0.15s ease;
    }
    .form-group input:focus, .form-group select:focus {
      border-color: var(--primary);
      outline: none;
      box-shadow: 0 0 0 2px rgba(10, 132, 255, 0.2);
    }
    .btn-save-cfg {
      background: var(--primary);
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.15);
      padding: 10px 22px;
      border-radius: var(--radius-sm);
      font-weight: 600;
      font-size: 0.88rem;
      cursor: pointer;
      margin-top: 10px;
      transition: all 0.15s ease;
      box-shadow: 0 2px 8px rgba(10, 132, 255, 0.3);
    }
    .btn-save-cfg:hover {
      background: var(--primary-hover);
      box-shadow: 0 4px 14px rgba(10, 132, 255, 0.4);
    }
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: var(--card-bg-elevated);
      color: var(--text-primary);
      border: 1px solid var(--card-border);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
      padding: 12px 20px;
      border-radius: var(--radius-md);
      font-weight: 500;
      font-size: 0.86rem;
      display: none;
      z-index: 200;
    }

    /* Guide Tab */
    .guide-banner {
      background: var(--card-bg-elevated);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 22px 26px;
      margin-bottom: 20px;
    }
    .guide-step-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .guide-step-card {
      background: var(--surface-inset);
      border: 1px solid var(--card-border-subtle);
      border-radius: var(--radius-md);
      padding: 18px;
    }
    .guide-step-num {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 26px;
      height: 26px;
      background: var(--primary);
      color: #fff;
      font-weight: 700;
      border-radius: 50%;
      font-size: 0.84rem;
      margin-right: 8px;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .guide-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    .guide-table th, .guide-table td {
      border: 1px solid var(--card-border);
      padding: 10px 14px;
      text-align: left;
      font-size: 0.84rem;
    }
    .guide-table th {
      background: var(--surface-inset);
      color: var(--text-muted);
      font-weight: 600;
    }
    .guide-table tr:nth-child(even) {
      background: rgba(255, 255, 255, 0.02);
    }
    .guide-faq-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }
    .guide-faq-card {
      background: var(--surface-inset);
      border: 1px solid var(--card-border-subtle);
      border-radius: var(--radius-md);
      padding: 16px 20px;
    }

    /* Interactive Spotlight Product Tour */
    .tour-overlay {
      position: fixed;
      inset: 0;
      z-index: 1000;
      background: transparent;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: auto;
      transition: opacity 0.25s ease;
    }
    .tour-spotlight-box {
      position: fixed;
      background: transparent;
      border-radius: var(--radius-md);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.78), 0 0 20px rgba(10, 132, 255, 0.45);
      border: 2px solid var(--primary);
      pointer-events: none;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 1001;
    }
    .tour-card {
      position: fixed;
      background: var(--card-bg);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: var(--radius-lg);
      padding: 16px 18px;
      width: 330px;
      max-width: 90vw;
      box-shadow: 0 20px 48px rgba(0, 0, 0, 0.75), 0 0 0 1px rgba(255, 255, 255, 0.08);
      z-index: 1002;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      animation: tourFadeIn 0.22s ease;
    }
    @keyframes tourFadeIn {
      from { opacity: 0; transform: translateY(6px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .tour-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
    }
    .tour-card-title-group {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .tour-step-badge {
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .tour-card-title {
      font-size: 0.92rem;
      font-weight: 600;
      color: var(--text-primary);
      letter-spacing: -0.01em;
    }
    .tour-btn-close {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.85rem;
      cursor: pointer;
      padding: 2px 6px;
      border-radius: 4px;
      line-height: 1;
      transition: all 0.15s ease;
    }
    .tour-btn-close:hover {
      color: #ffffff;
      background: rgba(255, 255, 255, 0.08);
    }
    .tour-card-desc {
      font-size: 0.82rem;
      color: var(--text);
      line-height: 1.45;
    }
    .tour-card-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid var(--card-border-subtle);
      padding-top: 8px;
      margin-top: 2px;
    }
    .tour-card-ref {
      font-size: 0.68rem;
      color: var(--text-faint);
      font-family: var(--font-mono);
    }
    .tour-btn-group {
      display: flex;
      gap: 6px;
      align-items: center;
      margin-left: auto;
    }
    .tour-btn {
      padding: 4px 10px;
      font-size: 0.76rem;
      font-weight: 500;
      border-radius: var(--radius-sm);
      background: var(--card-bg-elevated);
      color: var(--text);
      border: 1px solid var(--card-border);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .tour-btn:hover:not(:disabled) {
      background: #232e45;
      color: #fff;
      border-color: rgba(255, 255, 255, 0.18);
    }
    .tour-btn:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }
    .tour-btn-primary {
      background: var(--primary);
      color: #fff;
      border-color: rgba(255, 255, 255, 0.15);
      font-weight: 600;
    }
    .tour-btn-primary:hover:not(:disabled) {
      background: var(--primary-hover);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span>reBot Arm B601</span>
    </div>

    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('station')">Visual Grasping</button>
      <button class="tab-btn" onclick="switchTab('waypoints')">Waypoints Editor</button>
      <button class="tab-btn" onclick="switchTab('config')">Robot & Camera Config</button>
      <button class="tab-btn" onclick="switchTab('teach')">Teach &amp; Record</button>
    </div>

    <div class="header-actions">
      <span id="badge-status" class="status-badge status-live">LIVE</span>
      <div id="badge-enabled-container" class="robot-led-badge" title="Hardware Motor Driver Feedback">
        <span class="led-bulb led-off" id="led-indicator-bulb"></span>
        <span id="badge-enabled-text" style="color: var(--text-muted);">ROBOT DISABLED</span>
      </div>
      <span id="badge-waypoints" class="status-badge status-waypoints" title="Active Robot Waypoints (Click to view & edit)" onclick="switchTab('waypoints')">2 WPs (0 App / 2 Ret)</span>
      <button id="btn-header-guide" class="btn-guide-pill" onclick="startProductTour()" title="Start Interactive Spotlight Tour">[ ? Guide ]</button>
      <span id="badge-mode" style="font-size: 0.78rem; color: var(--text-faint); font-family: var(--font-mono);">Connected</span>
    </div>
  </header>

  <main>
    <!-- TAB 1: VISUAL GRASPING STATION -->
    <div id="tab-station" class="tab-content active">
      <div class="station-grid">
        
        <!-- Left: Live Video Feed -->
        <div>
          <div class="card" style="padding: 12px;">
            <div class="video-container">
              <img id="stream-img" src="/video_feed" alt="Live Stream">
            </div>
          </div>
        </div>

        <!-- Right: Interactive Controls & Queue -->
        <div>
          <!-- Tube Selection -->
          <div class="card">
            <div class="card-title">
              <span>DETECTED TUBES (LEFT TO RIGHT)</span>
              <span id="tubes-count" style="font-size: 0.78rem; color: var(--text-muted); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0 tubes</span>
            </div>
            <div id="tubes-bar" class="tube-buttons-bar">
              <span class="empty-queue-msg">Scanning table...</span>
            </div>
          </div>

          <!-- Picking Queue -->
          <div class="card">
            <div class="card-title">
              <span>PICKING QUEUE (CUSTOM ORDER)</span>
              <button onclick="clearQueue()" style="background:transparent; border:none; color:var(--danger); font-size:0.78rem; cursor:pointer; font-weight:600;">Clear All</button>
            </div>
            <div id="queue-box" class="queue-list">
              <span class="empty-queue-msg">No tubes queued. Click buttons above to add.</span>
            </div>

            <!-- Hardware Motor Driver Feedback Indicator Card -->
            <div class="card" style="padding: 12px 16px; margin-bottom: 14px; background: var(--surface-inset); border: 1px solid var(--card-border-subtle); border-left: 3px solid #475569;" id="card-led-border">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                  <span class="led-bulb led-off" id="card-led-bulb" style="width: 12px; height: 12px;"></span>
                  <div>
                    <div style="font-size: 0.7rem; color: var(--text-faint); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Motor Driver Feedback</div>
                    <div id="card-led-text" style="font-size: 0.88rem; font-weight: 600; color: #94a3b8; font-variant-numeric: tabular-nums;">Motors Disabled (Torque Off)</div>
                  </div>
                </div>
                <div id="card-led-subtext" style="font-size: 0.72rem; color: var(--text-faint); background: rgba(255,255,255,0.03); padding: 3px 8px; border-radius: var(--radius-sm); font-family: var(--font-mono);">Hardware feedback offline</div>
              </div>
            </div>

            <!-- Action Buttons -->
            <div class="actions-grid">
              <button id="btn-resume" class="btn-main btn-resume" onclick="resumeMotion()">
                RESUME FROM PAUSE
              </button>
              <button id="btn-grasp" class="btn-main btn-grasp" onclick="graspQueue()">
                GRASP QUEUE
              </button>
              <button id="btn-stop" class="btn-main btn-stop" onclick="emergencyStop()" title="Emergency Stop: Abort execution, open gripper, return to safe home, and disable motors">
                EMERGENCY STOP
              </button>
              <button id="btn-enable" class="btn-main btn-enable" onclick="enableRobot()">
                Enable Robot
              </button>
              <button id="btn-disable" class="btn-main btn-disable" onclick="disableRobot()">
                Disable Robot
              </button>
              <button class="btn-main btn-center" onclick="macroCenter()">
                Macro Center
              </button>
              <button class="btn-main btn-home" onclick="safeHome()">
                Safe Home
              </button>
              <button class="btn-main btn-ready" onclick="readyPose()">
                Ready Pose
              </button>
              <button class="btn-main btn-reset" onclick="unfreezeStream()">
                Reset / Unfreeze
              </button>
            </div>
          </div>

          <!-- System Telemetry & Error Log -->
          <div class="card">
            <div class="card-title" style="flex-wrap: wrap; gap: 8px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span>SYSTEM TELEMETRY &amp; DEBUG LOG</span>
                <span id="log-error-pill" class="log-pill log-pill-clean">0 Errors</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <div class="log-filter-group">
                  <button id="filter-all" class="btn-filter active" onclick="setLogFilter('all')">All</button>
                  <button id="filter-errors" class="btn-filter" onclick="setLogFilter('errors')">Errors (<span id="filter-err-count">0</span>)</button>
                </div>
                <label style="display:flex; align-items:center; gap:5px; font-size:0.75rem; color:var(--text-muted); cursor:pointer; user-select:none;">
                  <input type="checkbox" id="chk-autoscroll" checked style="width:auto; cursor:pointer;"> Auto-scroll
                </label>
                <button class="btn-sm" style="background:var(--card-bg-elevated); color:var(--text); border:1px solid var(--card-border);" onclick="copyAllLogs()" title="Copy all logs to clipboard">Copy</button>
                <button class="btn-sm" style="background:var(--surface-inset); color:var(--text-muted); border:1px solid var(--card-border);" onclick="clearLogs()" title="Clear logs">Clear</button>
                <button class="btn-sm" style="background:rgba(255,159,10,0.1); color:var(--warning); border:1px solid rgba(255,159,10,0.3);" onclick="triggerTestError()" title="Simulate an error for debugging test">Test Error</button>
              </div>
            </div>

            <!-- Active Error Banner (Prominent Alert for Debugging) -->
            <div id="active-error-banner" class="active-error-banner" style="display: none;">
              <div class="error-banner-header">
                <span class="error-banner-title">RUNTIME ERROR DETECTED</span>
                <span id="error-banner-time" class="error-banner-time"></span>
                <button class="error-banner-dismiss" onclick="dismissActiveError()" title="Dismiss banner">X</button>
              </div>
              <div id="error-banner-msg" class="error-banner-msg"></div>
              <details id="error-banner-trace-details" class="error-trace-details" style="display: none;">
                <summary>View Stack Traceback</summary>
                <pre id="error-banner-trace" class="error-trace-pre"></pre>
              </details>
            </div>

            <!-- Console Box -->
            <div id="console-log" class="console-box"></div>
          </div>
        </div>

      </div>
    </div>

    <!-- TAB 2: WAYPOINTS EDITOR -->
    <div id="tab-waypoints" class="tab-content">
      <div class="card">
        <div class="card-title">
          <span>MOTION WAYPOINTS EDITOR (APPROACH &amp; RETREAT)</span>
          <div style="display: flex; gap: 8px;">
            <button class="btn-sm" style="background:var(--card-bg-elevated); color:var(--text); border:1px solid var(--card-border);" onclick="loadWaypoints()">Reload</button>
            <button class="btn-sm" style="background:var(--primary); color:#fff; border:1px solid rgba(255,255,255,0.15);" onclick="saveWaypointsToDisk()">Save to waypoints.yaml</button>
          </div>
        </div>
        <div style="background: var(--surface-inset); border: 1px solid var(--card-border-subtle); border-radius: var(--radius-sm); padding: 10px 14px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div>
            <span style="font-weight: 600; color: var(--text-muted); font-size: 0.82rem;">Active Robot Configuration:</span>
            <span style="font-family: var(--font-mono); color: var(--cyan); margin-left: 6px; font-size: 0.84rem;" id="wp-cfg-filename">config/waypoints.yaml</span>
          </div>
          <div style="font-size: 0.78rem; color: var(--text-faint);">
            Waypoints shown below are the exact active via-points executed by the robot. Changes are saved and persist automatically.
          </div>
        </div>
        <p style="font-size: 0.84rem; color: var(--text-muted); margin-bottom: 16px;">
          Configure intermediate waypoints before grasping (Approach) and after grasping (Retreat). Position values can be Relative to Grasp Point or Absolute in Base Frame.
        </p>

        <!-- TWO COLUMN GRID: LEFT = APPROACH, RIGHT = RETREAT -->
        <div class="wp-editor-grid">
          
          <!-- Column 1: Approach Waypoints -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title" style="color: var(--cyan);">
              <span>APPROACH WAYPOINTS (READY POSE -> TARGET)</span>
              <span id="approach-count" style="font-size: 0.78rem; color: var(--text-muted); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0 waypoints</span>
            </div>
            
            <div id="approach-wps-list" class="wps-table-container"></div>

            <!-- Form to add an Approach Waypoint -->
            <div class="wp-form-box">
              <div class="wp-form-title">+ Add Approach Waypoint</div>
              
              <div class="form-group">
                <label>Coordinate Mode</label>
                <select id="app-mode" onchange="toggleWpCoordLabels('app')">
                  <option value="relative">Relative to Grasp [dx, dy, dz] (meters, m)</option>
                  <option value="absolute">Absolute Base Frame [X, Y, Z] (meters, m)</option>
                </select>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                <div class="form-group">
                  <label id="app-lbl-x">dx (m)</label>
                  <input type="number" step="0.005" id="app-pos-x" value="0.00">
                </div>
                <div class="form-group">
                  <label id="app-lbl-y">dy (m)</label>
                  <input type="number" step="0.005" id="app-pos-y" value="0.00">
                </div>
                <div class="form-group">
                  <label id="app-lbl-z">dz (m)</label>
                  <input type="number" step="0.005" id="app-pos-z" value="0.10">
                </div>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                <div class="form-group">
                  <label>Roll (rad)</label>
                  <input type="number" step="0.05" id="app-rot-roll" value="0.00">
                </div>
                <div class="form-group">
                  <label>Pitch (rad)</label>
                  <input type="number" step="0.05" id="app-rot-pitch" value="0.00">
                </div>
                <div class="form-group">
                  <label>Yaw (rad)</label>
                  <input type="number" step="0.05" id="app-rot-yaw" value="0.00">
                </div>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                <div class="form-group">
                  <label>Duration (seconds, s)</label>
                  <input type="number" step="0.2" id="app-duration" value="2.0">
                </div>
                <div class="form-group">
                  <label>Gripper Action</label>
                  <select id="app-gripper">
                    <option value="">None (Keep current state)</option>
                    <option value="open">Open (Ready width: 0.055m)</option>
                    <option value="close">Close (0.00m)</option>
                  </select>
                </div>
              </div>

              <div class="form-group" style="display:flex; align-items:center; gap:8px;">
                <input type="checkbox" id="app-pause" style="width:auto;">
                <label for="app-pause" style="margin-bottom:0; cursor:pointer;">Pause at Waypoint (waits for Resume click)</label>
              </div>

              <button class="btn-main" style="background:var(--primary); color:#fff; width:100%; padding:10px; border-color:rgba(255,255,255,0.15);" onclick="addWaypoint('approach')">
                + Append Approach Waypoint
              </button>
            </div>
          </div>

          <!-- Column 2: Retreat Waypoints -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title" style="color: var(--success);">
              <span>RETREAT WAYPOINTS (GRASPED -> READY POSE)</span>
              <span id="retreat-count" style="font-size: 0.78rem; color: var(--text-muted); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">2 waypoints</span>
            </div>

            <div id="retreat-wps-list" class="wps-table-container"></div>

            <!-- Form to add a Retreat Waypoint -->
            <div class="wp-form-box">
              <div class="wp-form-title">+ Add Retreat Waypoint</div>

              <div class="form-group">
                <label>Coordinate Mode</label>
                <select id="ret-mode" onchange="toggleWpCoordLabels('ret')">
                  <option value="relative">Relative to Grasp [dx, dy, dz] (meters, m)</option>
                  <option value="absolute">Absolute Base Frame [X, Y, Z] (meters, m)</option>
                </select>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                <div class="form-group">
                  <label id="ret-lbl-x">dx (m)</label>
                  <input type="number" step="0.005" id="ret-pos-x" value="0.00">
                </div>
                <div class="form-group">
                  <label id="ret-lbl-y">dy (m)</label>
                  <input type="number" step="0.005" id="ret-pos-y" value="0.00">
                </div>
                <div class="form-group">
                  <label id="ret-lbl-z">dz (m)</label>
                  <input type="number" step="0.005" id="ret-pos-z" value="0.10">
                </div>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                <div class="form-group">
                  <label>Roll (rad)</label>
                  <input type="number" step="0.05" id="ret-rot-roll" value="0.00">
                </div>
                <div class="form-group">
                  <label>Pitch (rad)</label>
                  <input type="number" step="0.05" id="ret-rot-pitch" value="0.00">
                </div>
                <div class="form-group">
                  <label>Yaw (rad)</label>
                  <input type="number" step="0.05" id="ret-rot-yaw" value="0.00">
                </div>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                <div class="form-group">
                  <label>Duration (seconds, s)</label>
                  <input type="number" step="0.2" id="ret-duration" value="2.0">
                </div>
                <div class="form-group">
                  <label>Gripper Action</label>
                  <select id="ret-gripper">
                    <option value="">None (Keep current state)</option>
                    <option value="open">Open (Ready width: 0.055m)</option>
                    <option value="close">Close (0.00m)</option>
                  </select>
                </div>
              </div>

              <div class="form-group" style="display:flex; align-items:center; gap:8px;">
                <input type="checkbox" id="ret-pause" style="width:auto;">
                <label for="ret-pause" style="margin-bottom:0; cursor:pointer;">Pause at Waypoint (waits for Resume click)</label>
              </div>

              <button class="btn-main" style="background:var(--card-bg-elevated); color:var(--text); width:100%; padding:10px; border-color:var(--card-border);" onclick="addWaypoint('retreat')">
                + Append Retreat Waypoint
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- TAB 3: CONFIGURATION -->
    <div id="tab-config" class="tab-content">
      <div class="card">
        <div class="card-title">SYSTEM CONFIGURATION (config/default.yaml)</div>
        <p style="font-size: 0.84rem; color: var(--text-muted); margin-bottom: 16px;">
          Adjust camera, table plane, hand-eye trims, and ready pose directly. Changes write directly to disk.
        </p>

        <div class="config-grid">
          <!-- Camera Settings -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">CAMERA & FOCUS</div>
            <div class="form-group">
              <label>Camera Index / Path (camera.id)</label>
              <input type="text" id="cfg-cam-id">
            </div>
            <div class="form-group">
              <label>Fixed Focus Value (camera.focus)</label>
              <input type="number" id="cfg-cam-focus">
            </div>
            <div class="form-group">
              <label>Autofocus Mode (camera.autofocus)</label>
              <select id="cfg-cam-autofocus">
                <option value="false">false (Fixed Focus)</option>
                <option value="true">true (Autofocus Enabled)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Intrinsics File</label>
              <input type="text" id="cfg-cam-intrinsics">
            </div>
          </div>

          <!-- Grasping Pipeline Settings -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">GRASP PLANE & OFFSETS</div>
            <div class="form-group">
              <label>Table Z Plane (meters, base frame)</label>
              <input type="number" step="0.001" id="cfg-table-z">
            </div>
            <div class="form-group">
              <label>Pre-Grasp Offset (meters)</label>
              <input type="number" step="0.001" id="cfg-pregrasp-offset">
            </div>
            <div class="form-group">
              <label>Insertion Depth (meters, X push)</label>
              <input type="number" step="0.001" id="cfg-insertion-depth">
            </div>
            <div class="form-group">
              <label>Grasp Z Offset / Depth (meters, vertical plunge)</label>
              <input type="number" step="0.001" id="cfg-grasp-z-offset" placeholder="-0.020">
              <div style="font-size: 0.72rem; color: var(--text-faint); margin-top: 3px;">
                Vertical plunge depth below table Z (negative descends deeper, e.g. -0.020m = -2cm).
              </div>
            </div>
          </div>

          <!-- Hand-Eye Calibration -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">HAND-EYE CALIBRATION</div>
            <div class="form-group">
              <label>Hand-Eye Result File</label>
              <input type="text" id="cfg-he-file">
            </div>
            <div class="form-group">
              <label>Base Compensation X (m)</label>
              <input type="number" step="0.002" id="cfg-he-comp-x">
            </div>
            <div class="form-group">
              <label>Base Compensation Y (m)</label>
              <input type="number" step="0.002" id="cfg-he-comp-y">
            </div>
            <div class="form-group">
              <label>Base Compensation Z (m)</label>
              <input type="number" step="0.002" id="cfg-he-comp-z">
            </div>
          </div>

          <!-- Ready Pose -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">READY POSE</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
              <div class="form-group">
                <label>X (m)</label>
                <input type="number" step="0.01" id="cfg-rp-x">
              </div>
              <div class="form-group">
                <label>Y (m)</label>
                <input type="number" step="0.01" id="cfg-rp-y">
              </div>
              <div class="form-group">
                <label>Z (m)</label>
                <input type="number" step="0.01" id="cfg-rp-z">
              </div>
              <div class="form-group">
                <label>Pitch (rad)</label>
                <input type="number" step="0.05" id="cfg-rp-pitch">
              </div>
            </div>
          </div>

          <!-- Gripper Configuration -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">GRIPPER CONFIGURATION</div>
            <div class="form-group">
              <label>Ready Open Width (meters, max stroke 0.09m)</label>
              <input type="number" step="0.001" min="0.0" max="0.09" id="cfg-gripper-ready-open" placeholder="0.055">
              <div style="font-size: 0.72rem; color: var(--text-faint); margin-top: 3px;">
                Width gripper opens to at Ready Pose and approach (0.055m = 5.5cm).
              </div>
            </div>
            <div class="form-group">
              <label>Fixed Close Width (meters, optional)</label>
              <input type="number" step="0.001" min="0.0" max="0.09" id="cfg-gripper-fixed-close" placeholder="0.030">
              <div style="font-size: 0.72rem; color: var(--text-faint); margin-top: 3px;">
                Fixed close target width (0.03m = 3.0cm). Leave blank to use adaptive YOLO detection width.
              </div>
            </div>
            <div class="form-group">
              <label>Close Offset Margin (meters)</label>
              <input type="number" step="0.001" min="0.0" max="0.05" id="cfg-gripper-close-offset" placeholder="0.010">
              <div style="font-size: 0.72rem; color: var(--text-faint); margin-top: 3px;">
                Subtracted from YOLO jaw width when closing adaptively (0.01m = 1.0cm).
              </div>
            </div>
            <button class="btn-main" style="width: 100%; margin-top: 6px; font-size: 0.82rem; padding: 8px 12px; border: 1px solid rgba(56, 189, 248, 0.3); color: var(--cyan); background: var(--surface-inset);" onclick="testGripperWidth()">
              Test Gripper Width Now
            </button>
          </div>
        </div>

        <div style="display: flex; align-items: center; gap: 14px; margin-top: 14px; flex-wrap: wrap;">
          <button class="btn-save-cfg" style="margin-top: 0;" onclick="saveConfiguration(false)">Save Configuration to default.yaml</button>
          <span id="cfg-sync-status" style="font-size: 0.84rem; color: var(--success); font-weight: 500;">Synced with default.yaml</span>
        </div>
      </div>
    </div>

    <!-- TAB: TEACH & RECORD (MANUAL ARM TEACHING & GRAVITY COMPENSATION) -->
    <div id="tab-teach" class="tab-content">
      <div style="display: grid; grid-template-columns: 1.15fr 1fr; gap: 20px; align-items: start;">
        
        <!-- LEFT BOARD: Teach Controls & Live Telemetry -->
        <div class="card" style="margin-bottom: 0;">
          <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
            <span>TEACH CONTROLS &amp; LIVE TELEMETRY</span>
            <span id="teach-status-badge" class="status-badge" style="background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid var(--card-border);">TEACH MODE INACTIVE</span>
          </div>

          <!-- Master Toggle & Safety Alert -->
          <div style="margin-bottom: 16px;">
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 8px;">
              <button id="btn-teach-toggle" class="btn-main" style="background: var(--warning); color: #000; font-weight: 700; padding: 10px 18px; border: none; font-size: 0.9rem;" onclick="toggleTeachMode()">
                Start Teach Mode (Gravity Compensation)
              </button>
            </div>
            <div style="background: rgba(255, 159, 10, 0.06); border: 1px solid rgba(255, 159, 10, 0.18); border-left: 3px solid var(--warning); padding: 9px 12px; font-size: 0.78rem; color: #fde68a; border-radius: var(--radius-sm); line-height: 1.45;">
              <strong>Manual Hand Guidance</strong>: When Teach Mode is active, the motors enter compliant floating mode with gravity feedforward (tau = g(q)). Move the arm smoothly by hand to your target pose. Stopping Teach Mode automatically executes Safe Home.
            </div>
          </div>

          <!-- Slide Mode Selector (Segmented Toggle) -->
          <div style="margin-bottom: 16px; background: var(--surface-inset); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
              <span style="font-size: 0.76rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Slide Mode Selector:</span>
              <span id="teach-mode-indicator" style="font-size: 0.76rem; color: var(--cyan); font-family: var(--font-mono); font-weight: 600;">ACTIVE: CARTESIAN (XYZ RPY)</span>
            </div>
            <div style="display: flex; gap: 4px; background: var(--bg); padding: 3px; border-radius: var(--radius-sm); border: 1px solid var(--card-border);">
              <button id="btn-mode-cartesian" class="tab-btn active" style="flex: 1; padding: 7px 10px; font-size: 0.8rem; font-weight: 600;" onclick="setTeachSlideMode('cartesian')">
                Cartesian Mode (XYZ RPY)
              </button>
              <button id="btn-mode-joints" class="tab-btn" style="flex: 1; padding: 7px 10px; font-size: 0.8rem; font-weight: 600;" onclick="setTeachSlideMode('joints')">
                Joint Angles Mode (J1-J6)
              </button>
            </div>
            <div style="font-size: 0.72rem; color: var(--text-faint); margin-top: 6px;">
              Waypoints recorded will be saved in the format matching whichever mode is selected above.
            </div>
          </div>

          <!-- Reactive Live Telemetry HUD -->
          <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
              <span style="font-size: 0.76rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Live Arm Telemetry</span>
              <span id="teach-gripper-display" style="font-size: 0.78rem; color: var(--success); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">Gripper: 5.5 cm</span>
            </div>

            <!-- Cartesian Telemetry Card (Shown when Cartesian mode selected) -->
            <div id="teach-hud-cartesian" style="display: block;">
              <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px;">
                <div style="background: var(--surface-inset); padding: 9px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">X POSITION</div>
                  <div id="hud-val-x" style="font-size: 1.15rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 3px;">+0.3500 m</div>
                </div>
                <div style="background: var(--surface-inset); padding: 9px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">Y POSITION</div>
                  <div id="hud-val-y" style="font-size: 1.15rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 3px;">+0.0000 m</div>
                </div>
                <div style="background: var(--surface-inset); padding: 9px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">Z POSITION</div>
                  <div id="hud-val-z" style="font-size: 1.15rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 3px;">+0.2800 m</div>
                </div>
              </div>
              <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                <div style="background: var(--surface-inset); padding: 9px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">ROLL (X-AXIS)</div>
                  <div id="hud-val-roll-deg" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--text-primary); margin-top: 2px;">0.0°</div>
                  <div id="hud-val-roll" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.000 rad</div>
                </div>
                <div style="background: var(--surface-inset); padding: 9px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">PITCH (Y-AXIS)</div>
                  <div id="hud-val-pitch-deg" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--text-primary); margin-top: 2px;">34.4°</div>
                  <div id="hud-val-pitch" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.600 rad</div>
                </div>
                <div style="background: var(--surface-inset); padding: 9px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">YAW (Z-AXIS)</div>
                  <div id="hud-val-yaw-deg" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--text-primary); margin-top: 2px;">0.0°</div>
                  <div id="hud-val-yaw" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.000 rad</div>
                </div>
              </div>
            </div>

            <!-- Joint Angles Telemetry Card (Shown when Joint mode selected) -->
            <div id="teach-hud-joints" style="display: none;">
              <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px;">
                <div style="background: var(--surface-inset); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">JOINT 1 (BASE)</div>
                  <div id="hud-val-j1" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 2px;">0.0°</div>
                  <div id="hud-val-j1-rad" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.000 rad</div>
                </div>
                <div style="background: var(--surface-inset); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">JOINT 2 (SHOULDER)</div>
                  <div id="hud-val-j2" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 2px;">-37.2°</div>
                  <div id="hud-val-j2-rad" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">-0.650 rad</div>
                </div>
                <div style="background: var(--surface-inset); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">JOINT 3 (ELBOW)</div>
                  <div id="hud-val-j3" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 2px;">-48.7°</div>
                  <div id="hud-val-j3-rad" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">-0.850 rad</div>
                </div>
              </div>
              <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                <div style="background: var(--surface-inset); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">JOINT 4 (WRIST 1)</div>
                  <div id="hud-val-j4" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 2px;">0.0°</div>
                  <div id="hud-val-j4-rad" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.000 rad</div>
                </div>
                <div style="background: var(--surface-inset); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">JOINT 5 (WRIST 2)</div>
                  <div id="hud-val-j5" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 2px;">25.8°</div>
                  <div id="hud-val-j5-rad" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.450 rad</div>
                </div>
                <div style="background: var(--surface-inset); padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle); text-align: center;">
                  <div style="font-size: 0.68rem; color: var(--text-faint); font-weight: 600; letter-spacing: 0.04em;">JOINT 6 (FLANGE)</div>
                  <div id="hud-val-j6" style="font-size: 1.05rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 600; color: var(--cyan); margin-top: 2px;">0.0°</div>
                  <div id="hud-val-j6-rad" style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">0.000 rad</div>
                </div>
              </div>
            </div>

            <!-- Alternate Mode Preview Summary -->
            <div id="teach-hud-alt-summary" style="margin-top: 8px; font-size: 0.74rem; color: var(--text-muted); font-family: var(--font-mono); font-variant-numeric: tabular-nums; background: var(--surface-inset); padding: 5px 10px; border-radius: var(--radius-sm); border: 1px solid var(--card-border-subtle);">
              Joints: [0.0°, -37.2°, -48.7°, 0.0°, 25.8°, 0.0°]
            </div>
          </div>

          <!-- Waypoint Snapshot Controls -->
          <div style="background: var(--surface-inset); border: 1px solid var(--card-border-subtle); padding: 14px; border-radius: var(--radius-sm);">
            <div style="font-size: 0.78rem; font-weight: 600; color: var(--text-muted); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em;">
              Capture Waypoint Snapshot
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
              <div>
                <label style="font-size: 0.74rem; color: var(--text-muted); display: block; margin-bottom: 4px;">TRAJECTORY DURATION (SEC)</label>
                <input type="number" id="teach-record-duration" value="2.0" step="0.1" min="0.5" max="15.0" style="width: 100%; box-sizing: border-box; background: var(--bg); border: 1px solid var(--card-border); color: #fff; padding: 8px 10px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-variant-numeric: tabular-nums;">
              </div>
              <div>
                <label style="font-size: 0.74rem; color: var(--text-muted); display: block; margin-bottom: 4px;">GRIPPER STATE AT POSE</label>
                <select id="teach-record-gripper" style="width: 100%; box-sizing: border-box; background: var(--bg); border: 1px solid var(--card-border); color: #fff; padding: 8px 10px; border-radius: var(--radius-sm);">
                  <option value="keep">Keep Current State</option>
                  <option value="open">Open (Ready Open Width)</option>
                  <option value="close">Close (Clamped)</option>
                </select>
              </div>
            </div>
            <button id="btn-record-pose" class="btn-main" style="width: 100%; background: var(--primary); color: #fff; font-weight: 600; padding: 10px; border: 1px solid rgba(255,255,255,0.15); font-size: 0.9rem;" onclick="recordTeachPose()">
              Record Current Pose (saves as Cartesian)
            </button>
          </div>
        </div>

        <!-- RIGHT BOARD: Recorded Waypoints Board (Inspection & Staging) -->
        <div class="card" style="margin-bottom: 0;">
          <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
            <span>RECORDED WAYPOINTS BOARD</span>
            <span id="teach-staged-count" class="status-badge" style="background: rgba(10, 132, 255, 0.12); color: #60a5fa; border: 1px solid rgba(10, 132, 255, 0.28);">0 STAGED</span>
          </div>

          <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 12px; line-height: 1.45;">
            Points recorded during manual guidance appear below. Review each pose before committing to the robot's active waypoints. You may discard unwanted poses or keep them.
          </div>

          <!-- Staged Items List -->
          <div id="teach-staged-container" style="max-height: 420px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; padding-right: 4px;">
            <div id="teach-staged-empty" style="text-align: center; padding: 40px 16px; color: var(--text-faint); font-size: 0.82rem; border: 1px dashed var(--card-border); border-radius: var(--radius-sm);">
              No waypoints recorded yet.<br>Move arm to desired target pose and click "Record Current Pose".
            </div>
          </div>

          <!-- Staging Board Export & Action Controls -->
          <div style="border-top: 1px solid var(--card-border); padding-top: 14px;">
            <div style="font-size: 0.76rem; font-weight: 600; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;">
              Export Staged Points:
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;">
              <button class="btn-main" style="font-size: 0.82rem; padding: 8px;" onclick="commitTeachWaypoints('approach')">
                Add to Approach Waypoints
              </button>
              <button class="btn-main" style="font-size: 0.82rem; padding: 8px;" onclick="commitTeachWaypoints('retreat')">
                Add to Retreat Waypoints
              </button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
              <button class="btn-main" style="font-size: 0.82rem; padding: 8px;" onclick="commitTeachWaypoints('ready_pose')">
                Set as Ready Pose
              </button>
              <button class="btn-main" style="background: rgba(255, 69, 58, 0.1); color: var(--danger); border-color: rgba(255, 69, 58, 0.25); font-size: 0.82rem; padding: 8px;" onclick="clearTeachStaging()">
                Discard All
              </button>
            </div>
          </div>

        </div>

      </div>
    </div>

    <!-- INTERACTIVE SPOTLIGHT PRODUCT TOUR OVERLAY -->
    <div id="tour-overlay" class="tour-overlay" style="display: none;" onclick="closeProductTour()">
      <div id="tour-spotlight-box" class="tour-spotlight-box"></div>
      <div id="tour-card" class="tour-card" onclick="event.stopPropagation()">
        <div class="tour-card-header">
          <div class="tour-card-title-group">
            <span id="tour-card-step-badge" class="tour-step-badge">Step 1 of 6</span>
            <div id="tour-card-title" class="tour-card-title">Live Camera Feed</div>
          </div>
          <button class="tour-btn-close" onclick="closeProductTour()" title="Close Tour">X</button>
        </div>
        <p id="tour-card-desc" class="tour-card-desc">
          HOW TO PICK TUBES IN 3 STEPS: View the live stream. Falcon tubes are sorted and numbered from left to right.
        </p>
        <div class="tour-card-footer">
          <span class="tour-card-ref">User Guide &bull; docs/WEB_GUI_USER_GUIDE.md</span>
          <div class="tour-btn-group">
            <button id="tour-btn-back" class="tour-btn" onclick="prevTourStep()">Back</button>
            <button id="tour-btn-next" class="tour-btn tour-btn-primary" onclick="nextTourStep()">Next</button>
            <button class="tour-btn" onclick="closeProductTour()">Close</button>
          </div>
        </div>
      </div>
    </div>
  </main>

  <div id="toast" class="toast">Configuration saved successfully</div>


  <script>
    let currentState = 'live';
    let currentTubes = [];
    let currentQueue = [];

    let currentTab = 'station';
    function switchTab(tabId) {
      if (tabId === 'guide') {
        startProductTour();
        return;
      }
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      try { sessionStorage.setItem('active_tab', tabId); } catch(e) {}
      currentTab = tabId;
      if (tabId === 'station') {
        const btn = document.querySelector('.nav-tabs .tab-btn:nth-child(1)');
        if (btn) btn.classList.add('active');
        const tab = document.getElementById('tab-station');
        if (tab) tab.classList.add('active');
      } else if (tabId === 'waypoints') {
        const btn = document.querySelector('.nav-tabs .tab-btn:nth-child(2)');
        if (btn) btn.classList.add('active');
        const tab = document.getElementById('tab-waypoints');
        if (tab) tab.classList.add('active');
        loadWaypoints();
      } else if (tabId === 'config') {
        const btn = document.querySelector('.nav-tabs .tab-btn:nth-child(3)');
        if (btn) btn.classList.add('active');
        const tab = document.getElementById('tab-config');
        if (tab) tab.classList.add('active');
        loadConfiguration();
      } else if (tabId === 'teach') {
        const btn = document.querySelector('.nav-tabs .tab-btn:nth-child(4)');
        if (btn) btn.classList.add('active');
        const tab = document.getElementById('tab-teach');
        if (tab) tab.classList.add('active');
        fetchTeachPose();
      }
    }

    // ==========================================
    // Interactive Spotlight Product Tour
    // ==========================================
    const TOUR_DATA = {
      station: [
        {
          target: '.video-container',
          title: 'Live Camera Feed',
          desc: 'HOW TO PICK TUBES IN 3 STEPS: View the live stream. Falcon tubes are sorted and numbered from left to right.'
        },
        {
          target: '#tubes-bar',
          title: 'Tube Buttons Bar',
          desc: 'Click any numbered tube button to add it to your custom sequential picking queue.'
        },
        {
          target: '#queue-box',
          title: 'Picking Queue',
          desc: 'Review queued tubes in execution order. Click the remove button to discard unwanted items.'
        },
        {
          target: '#card-led-border',
          title: 'Motor Driver Feedback',
          desc: 'Live hardware torque status. Green indicates motors are powered and active.'
        },
        {
          target: '.actions-grid',
          title: 'Master Motion Controls',
          desc: 'Click GRASP QUEUE to execute picking, or EMERGENCY STOP to abort motion immediately.'
        },
        {
          target: '#console-log',
          title: 'System Telemetry Log',
          desc: 'Live hardware events and warnings. Filter, copy, or clear logs at any time.'
        }
      ],
      waypoints: [
        {
          target: '#wp-cfg-filename',
          title: 'Active Configuration',
          desc: 'Displays the active waypoints file. Changes save directly to waypoints.yaml on disk.'
        },
        {
          target: '#approach-wps-list',
          title: 'Approach Waypoints',
          desc: 'Via-points executed before grasping. Configure relative offsets or absolute coordinates.'
        },
        {
          target: '#retreat-wps-list',
          title: 'Retreat Waypoints',
          desc: 'Via-points executed after grasping to safely clear the rack before homing.'
        },
        {
          target: '#app-mode',
          title: 'Waypoint Form',
          desc: 'Append custom waypoints using relative offsets or absolute coordinates with optional pause.'
        }
      ],
      config: [
        {
          target: '#cfg-cam-id',
          title: 'Camera & Focus Settings',
          desc: 'Set camera device ID, fixed manual focus value, and optical intrinsics file path.'
        },
        {
          target: '#cfg-table-z',
          title: 'Grasp Plane & Offsets',
          desc: 'Calibrate table height, pregrasp elevation, insertion depth, and vertical plunge depth.'
        },
        {
          target: '#cfg-he-file',
          title: 'Hand-Eye Calibration',
          desc: 'Select hand-eye matrix file and calibrate translational compensation offsets in meters.'
        },
        {
          target: '#cfg-rp-x',
          title: 'Ready Pose Settings',
          desc: 'Configure the default camera observation position in base frame coordinates.'
        },
        {
          target: '#cfg-gripper-ready-open',
          title: 'Gripper Configuration',
          desc: 'Set ready opening width, fixed close targets, and test clamping width.'
        },
        {
          target: '.btn-save-cfg',
          title: 'Save Configuration',
          desc: 'Writes all updated parameters directly into default.yaml and syncs robot state.'
        }
      ],
      teach: [
        {
          target: '#btn-teach-toggle',
          title: 'Gravity Compensation',
          desc: 'Activates compliant zero-torque floating mode. Hand-guide the arm smoothly to desired target poses.'
        },
        {
          target: '#btn-mode-cartesian',
          title: 'Slide Mode Selector',
          desc: 'Toggle recording format between Cartesian XYZ RPY coordinates and 6-axis joint angles.'
        },
        {
          target: '#teach-hud-cartesian',
          title: 'Live Arm Telemetry HUD',
          desc: 'Streams live coordinates or joint angles in real time with tabular precision.'
        },
        {
          target: '#btn-record-pose',
          title: 'Capture Snapshot',
          desc: 'Set duration and gripper action, then click Record to stage the current arm pose.'
        },
        {
          target: '#teach-staged-container',
          title: 'Recorded Waypoints Board',
          desc: 'Review and inspect staged poses. Discard unwanted points before exporting.'
        },
        {
          target: '#teach-staged-count',
          title: 'Export Staged Points',
          desc: 'Export staged points to Approach waypoints, Retreat waypoints, or Ready Pose.'
        }
      ]
    };

    let tourActive = false;
    let tourStepIndex = 0;
    let tourSteps = [];

    function startProductTour() {
      const activeTabId = currentTab || 'station';
      tourSteps = TOUR_DATA[activeTabId] || TOUR_DATA['station'];
      if (!tourSteps || tourSteps.length === 0) return;
      tourActive = true;
      tourStepIndex = 0;
      const overlay = document.getElementById('tour-overlay');
      if (overlay) overlay.style.display = 'flex';
      renderTourStep();
    }

    function closeProductTour() {
      tourActive = false;
      const overlay = document.getElementById('tour-overlay');
      if (overlay) overlay.style.display = 'none';
    }

    function prevTourStep() {
      if (!tourActive || tourStepIndex <= 0) return;
      tourStepIndex--;
      renderTourStep();
    }

    function nextTourStep() {
      if (!tourActive) return;
      if (tourStepIndex >= tourSteps.length - 1) {
        closeProductTour();
        return;
      }
      tourStepIndex++;
      renderTourStep();
    }

    function renderTourStep() {
      if (!tourActive || !tourSteps || tourSteps.length === 0) return;
      const step = tourSteps[tourStepIndex];
      const targetEl = document.querySelector(step.target);

      // Update text and indicators
      const badge = document.getElementById('tour-card-step-badge');
      if (badge) badge.innerText = `Step ${tourStepIndex + 1} of ${tourSteps.length}`;
      const title = document.getElementById('tour-card-title');
      if (title) title.innerText = step.title;
      const desc = document.getElementById('tour-card-desc');
      if (desc) desc.innerText = step.desc;

      const backBtn = document.getElementById('tour-btn-back');
      if (backBtn) backBtn.disabled = (tourStepIndex === 0);

      const nextBtn = document.getElementById('tour-btn-next');
      if (nextBtn) {
        nextBtn.innerText = (tourStepIndex === tourSteps.length - 1) ? 'Finish' : 'Next';
      }

      // Position spotlight box and floating card
      const spotlight = document.getElementById('tour-spotlight-box');
      const card = document.getElementById('tour-card');

      if (targetEl && spotlight && card) {
        const updateSpotlight = () => {
          const rect = targetEl.getBoundingClientRect();
          const pad = 6;
          spotlight.style.top = `${Math.max(0, rect.top - pad)}px`;
          spotlight.style.left = `${Math.max(0, rect.left - pad)}px`;
          spotlight.style.width = `${rect.width + pad * 2}px`;
          spotlight.style.height = `${rect.height + pad * 2}px`;

          // Position card adjacent to target
          const cardWidth = 340;
          const cardHeight = 160;
          let cardTop = rect.bottom + 14;
          let cardLeft = rect.left + (rect.width / 2) - (cardWidth / 2);

          // If overflowing bottom, position above
          if (cardTop + cardHeight > window.innerHeight - 20) {
            cardTop = Math.max(20, rect.top - cardHeight - 14);
          }
          // Clamp horizontally
          cardLeft = Math.max(16, Math.min(window.innerWidth - cardWidth - 16, cardLeft));

          card.style.top = `${cardTop}px`;
          card.style.left = `${cardLeft}px`;
        };

        targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        updateSpotlight();
        setTimeout(updateSpotlight, 100);
      }
    }

    // Keyboard navigation for Tour
    window.addEventListener('keydown', (e) => {
      if (!tourActive) return;
      if (e.key === 'Escape') closeProductTour();
      else if (e.key === 'ArrowRight') nextTourStep();
      else if (e.key === 'ArrowLeft') prevTourStep();
    });

    window.addEventListener('resize', () => {
      if (tourActive) renderTourStep();
    });
    window.addEventListener('scroll', () => {
      if (tourActive) renderTourStep();
    }, true);

    // Polling System State
    async function updateState() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        currentState = data.status;
        currentTubes = data.tubes || [];
        currentQueue = data.queue || [];

        // Update Status Badge
        const badge = document.getElementById('badge-status');
        badge.className = 'status-badge status-' + data.status;
        badge.innerText = data.status.toUpperCase();

        // Update Enabled Badge & Hardware LED Indicator
        const isEnabled = Boolean(data.robot_enabled);
        const isDryRun = Boolean(data.dry_run);
        const isConnected = Boolean(data.robot_connected);

        const bulb = document.getElementById('led-indicator-bulb');
        const badgeText = document.getElementById('badge-enabled-text');
        const badgeContainer = document.getElementById('badge-enabled-container');
        const cardBulb = document.getElementById('card-led-bulb');
        const cardText = document.getElementById('card-led-text');
        const cardSub = document.getElementById('card-led-subtext');
        const cardBorder = document.getElementById('card-led-border');

        if (isEnabled) {
          // Green glowing LED (Driver Hardware Feedback Confirmed)
          if (bulb) bulb.className = 'led-bulb led-on';
          if (badgeText) {
            badgeText.innerText = 'ROBOT ENABLED';
            badgeText.style.color = '#10b981';
          }
          if (badgeContainer) {
            badgeContainer.style.borderColor = 'rgba(16, 185, 129, 0.4)';
            badgeContainer.style.background = 'rgba(16, 185, 129, 0.1)';
          }
          if (cardBulb) cardBulb.className = 'led-bulb led-on';
          if (cardText) {
            cardText.innerText = 'Motors Enabled (Torque Active)';
            cardText.style.color = '#10b981';
          }
          if (cardSub) {
            cardSub.innerText = isDryRun ? 'Simulated Driver Confirmed' : 'Hardware Feedback OK (Holding Pose)';
            cardSub.style.color = '#34d399';
          }
          if (cardBorder) cardBorder.style.borderLeftColor = '#10b981';
        } else {
          // Dim Gray LED (Torque Off / Driver Disabled / No Feedback)
          if (bulb) bulb.className = 'led-bulb led-off';
          if (badgeText) {
            badgeText.innerText = 'ROBOT DISABLED';
            badgeText.style.color = '#94a3b8';
          }
          if (badgeContainer) {
            badgeContainer.style.borderColor = 'var(--card-border)';
            badgeContainer.style.background = 'rgba(15, 23, 42, 0.7)';
          }
          if (cardBulb) cardBulb.className = 'led-bulb led-off';
          if (cardText) {
            cardText.innerText = 'Motors Disabled (Torque Off)';
            cardText.style.color = '#94a3b8';
          }
          if (cardSub) {
            cardSub.innerText = isConnected ? 'Torque cut (Free to move by hand)' : 'Hardware disconnected or offline';
            cardSub.style.color = '#64748b';
          }
          if (cardBorder) cardBorder.style.borderLeftColor = '#475569';
        }

        document.getElementById('badge-mode').innerText = data.dry_run ? 'Simulation Mode' : (data.robot_connected ? 'Hardware Connected' : 'Simulated');

        // Resume Button visibility
        const resumeBtn = document.getElementById('btn-resume');
        if (data.status === 'paused') {
          resumeBtn.classList.add('active');
        } else {
          resumeBtn.classList.remove('active');
        }

        // Grasp button disabled state
        document.getElementById('btn-grasp').disabled = (currentQueue.length === 0 || data.status === 'executing');

        // Render Tube Buttons
        renderTubesBar(currentTubes, currentQueue);

        // Render Queue
        renderQueue(currentQueue);

        // Update Error Count & Pill
        const errCount = data.error_count || 0;
        const errPill = document.getElementById('log-error-pill');
        const filterErrCount = document.getElementById('filter-err-count');
        if (filterErrCount) filterErrCount.innerText = errCount;
        if (errPill) {
          if (errCount > 0) {
            errPill.innerText = `${errCount} Error${errCount === 1 ? '' : 's'}`;
            errPill.className = 'log-pill log-pill-alert';
          } else {
            errPill.innerText = '0 Errors';
            errPill.className = 'log-pill log-pill-clean';
          }
        }

        // Active Error Banner
        const errBanner = document.getElementById('active-error-banner');
        if (data.last_error) {
          errBanner.style.display = 'block';
          document.getElementById('error-banner-time').innerText = data.last_error.time || '';
          document.getElementById('error-banner-msg').innerText = data.last_error.text || 'An unknown runtime error occurred.';
          const tbEl = document.getElementById('error-banner-trace');
          const tbDetails = document.getElementById('error-banner-trace-details');
          if (data.last_error.traceback) {
            tbEl.innerText = data.last_error.traceback;
            tbDetails.style.display = 'block';
          } else {
            tbDetails.style.display = 'none';
          }
        } else {
          errBanner.style.display = 'none';
        }

        // Update Waypoints Badge
        const appCnt = data.approach_count !== undefined ? data.approach_count : currentApproachWps.length;
        const retCnt = data.retreat_count !== undefined ? data.retreat_count : currentRetreatWps.length;
        updateWaypointsBadge(appCnt, retCnt);

        // Render Console Logs
        renderLogs(data.logs || []);

      } catch (err) {
        console.error("Status poll error", err);
      }
    }

    let lastTubesBarHtml = null;
    function renderTubesBar(tubes, queue) {
      const container = document.getElementById('tubes-bar');
      const countEl = document.getElementById('tubes-count');
      if (countEl) countEl.innerText = `${tubes.length} tube${tubes.length === 1 ? '' : 's'} detected`;
      if (tubes.length === 0) {
        const emptyHtml = `<span class="empty-queue-msg">Scanning table...</span>`;
        if (lastTubesBarHtml !== emptyHtml) {
          container.innerHTML = emptyHtml;
          lastTubesBarHtml = emptyHtml;
        }
        return;
      }

      const queuedIds = new Set(queue.map(q => q.tube_id));
      const newHtml = tubes.map(t => {
        const isQueued = queuedIds.has(t.id);
        const cls = isQueued ? 'tube-btn queued' : 'tube-btn';
        const title = isQueued ? `Tube #${t.id} (In Queue - click to add another)` : `Click to add Tube #${t.id} to picking queue`;
        return `<button type="button" class="${cls}" data-tube-id="${t.id}" onclick="enqueueTube(${t.id})" title="${title}">#${t.id}</button>`;
      }).join('');

      // ONLY mutate DOM when buttons or queued status actually change.
      // This prevents destroying active button DOM elements mid-click during the 200ms poll cycle!
      if (newHtml !== lastTubesBarHtml) {
        container.innerHTML = newHtml;
        lastTubesBarHtml = newHtml;
      }
    }

    let lastQueueHtml = null;
    function renderQueue(queue) {
      const container = document.getElementById('queue-box');
      if (queue.length === 0) {
        const emptyHtml = `<span class="empty-queue-msg">No tubes queued. Click buttons above to add.</span>`;
        if (lastQueueHtml !== emptyHtml) {
          container.innerHTML = emptyHtml;
          lastQueueHtml = emptyHtml;
        }
        return;
      }
      const newHtml = queue.map((item, idx) => {
        return `
          <div class="queue-item">
            <span class="queue-item-info">${idx + 1}. ${item.label}</span>
            <span class="queue-item-coord">Base: (${item.x}, ${item.y}, ${item.z})</span>
            <button type="button" class="queue-item-remove" onclick="dequeueItem(${item.queue_uid})" title="Remove item">X</button>
          </div>
        `;
      }).join('');

      if (newHtml !== lastQueueHtml) {
        container.innerHTML = newHtml;
        lastQueueHtml = newHtml;
      }
    }

    let isEnqueuing = false;
    async function enqueueTube(tubeId) {
      if (isEnqueuing) return;
      isEnqueuing = true;

      // Instant optimistic UI feedback: visually depress and mark queued in 0ms
      const btn = document.querySelector(`.tube-btn[data-tube-id="${tubeId}"]`);
      if (btn) {
        btn.classList.add('queued');
        btn.style.transform = 'scale(0.92)';
        setTimeout(() => { if (btn) btn.style.transform = ''; }, 120);
      }

      try {
        await fetch('/api/queue', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'add', tube_id: tubeId })
        });
      } catch (err) {
        console.error("Failed to enqueue tube:", err);
      } finally {
        setTimeout(() => { isEnqueuing = false; }, 80);
      }
      updateState();
    }

    async function dequeueItem(queueUid) {
      await fetch('/api/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'remove', queue_uid: queueUid })
      });
      updateState();
    }

    async function clearQueue() {
      await fetch('/api/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear' })
      });
      updateState();
    }

    async function graspQueue() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'grasp' })
      });
      updateState();
    }

    async function emergencyStop() {
      const btn = document.getElementById('btn-stop');
      if (btn) {
        btn.style.transform = 'scale(0.96)';
        setTimeout(() => { if (btn) btn.style.transform = ''; }, 150);
      }
      try {
        await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'emergency_stop' })
        });
      } catch (err) {
        console.error("Failed to send emergency stop command:", err);
      }
      setTimeout(updateState, 100);
    }

    async function macroCenter() {
      const firstQueued = currentQueue.length > 0 ? currentQueue[0].tube_id : (currentTubes.length > 0 ? currentTubes[0].id : null);
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'macro_center', tube_id: firstQueued })
      });
      updateState();
    }

    async function safeHome() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'safe_home' })
      });
      updateState();
    }

    async function readyPose() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'ready_pose' })
      });
      updateState();
    }

    async function unfreezeStream() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'unfreeze' })
      });
      updateState();
    }

    async function resumeMotion() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'resume' })
      });
      updateState();
    }

    async function enableRobot() {
      // Set pending state (amber pulsing LED) while waiting for hardware driver verification
      const bulb = document.getElementById('led-indicator-bulb');
      const cardBulb = document.getElementById('card-led-bulb');
      const badgeText = document.getElementById('badge-enabled-text');
      const cardText = document.getElementById('card-led-text');
      const cardSub = document.getElementById('card-led-subtext');
      const cardBorder = document.getElementById('card-led-border');

      if (bulb) bulb.className = 'led-bulb led-pending';
      if (cardBulb) cardBulb.className = 'led-bulb led-pending';
      if (badgeText) {
        badgeText.innerText = 'VERIFYING FEEDBACK...';
        badgeText.style.color = '#f59e0b';
      }
      if (cardText) {
        cardText.innerText = 'Awaiting Driver Response...';
        cardText.style.color = '#f59e0b';
      }
      if (cardSub) cardSub.innerText = 'Sending CAN enable frames...';
      if (cardBorder) cardBorder.style.borderLeftColor = '#f59e0b';

      try {
        await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'enable_robot' })
        });
      } catch (err) {
        console.error('Failed to send enable command:', err);
      }
      setTimeout(updateState, 120);
    }

    async function disableRobot() {
      try {
        await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'disable_robot' })
        });
      } catch (err) {
        console.error('Failed to send disable command:', err);
      }
      setTimeout(updateState, 100);
    }

    // ==========================================
    // Telemetry & Debugging Log Functions
    // ==========================================
    let currentLogFilter = 'all'; // 'all' | 'errors'
    let cachedLogs = [];

    function setLogFilter(filter) {
      currentLogFilter = filter;
      const btnAll = document.getElementById('filter-all');
      const btnErr = document.getElementById('filter-errors');
      if (btnAll) btnAll.classList.toggle('active', filter === 'all');
      if (btnErr) btnErr.classList.toggle('active', filter === 'errors');
      renderLogs(cachedLogs);
    }

    function renderLogs(logs) {
      cachedLogs = logs || [];
      const logBox = document.getElementById('console-log');
      if (!logBox) return;

      const filtered = (currentLogFilter === 'errors')
        ? cachedLogs.filter(l => {
            const lvl = (typeof l === 'object' && l.level) ? l.level : '';
            return lvl === 'error' || lvl === 'warning';
          })
        : cachedLogs;

      if (filtered.length === 0) {
        logBox.innerHTML = (currentLogFilter === 'errors')
          ? '<div style="color:var(--text-muted); font-style:italic; padding:10px;">No errors or warnings recorded.</div>'
          : '<div style="color:var(--text-muted); font-style:italic; padding:10px;">No telemetry logs available.</div>';
        return;
      }

      logBox.innerHTML = filtered.map(item => {
        let text = typeof item === 'object' ? (item.text || item.message || '') : String(item);
        let time = typeof item === 'object' ? (item.time || '') : '';
        let level = typeof item === 'object' ? (item.level || 'info') : 'info';
        let trace = typeof item === 'object' ? (item.traceback || '') : '';

        // Extract [HH:MM:SS] if present
        if (!time) {
          const match = text.match(/^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$/);
          if (match) {
            time = match[1];
            text = match[2];
          }
        } else if (text.startsWith(`[${time}] `)) {
          text = text.substring(time.length + 3);
        }

        const badgeLabel = level === 'error' ? 'ERR' : (level === 'warning' ? 'WARN' : (level === 'success' ? 'OK' : 'INFO'));
        const traceHtml = trace
          ? `<details class="log-trace-details"><summary>View Stack Trace</summary><pre class="log-trace-pre">${escapeHtml(trace)}</pre></details>`
          : '';

        return `
          <div class="log-line log-${level}">
            <div class="log-row-main">
              <span class="log-time">${time}</span>
              <span class="log-badge">${badgeLabel}</span>
              <span class="log-msg">${escapeHtml(text)}</span>
            </div>
            ${traceHtml}
          </div>
        `;
      }).join('');

      const autoScrollChk = document.getElementById('chk-autoscroll');
      if (autoScrollChk && autoScrollChk.checked) {
        logBox.scrollTop = logBox.scrollHeight;
      }
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    async function clearLogs() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear_logs' })
      });
      updateState();
    }

    async function dismissActiveError() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'dismiss_error' })
      });
      const b = document.getElementById('active-error-banner');
      if (b) b.style.display = 'none';
      updateState();
    }

    async function triggerTestError() {
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'trigger_test_error' })
      });
      updateState();
    }

    function copyAllLogs() {
      const textToCopy = (cachedLogs || []).map(l => {
        if (typeof l === 'object') {
          let line = `[${l.time}] [${(l.level || 'info').toUpperCase()}] ${l.text || l.message || ''}`;
          if (l.traceback) {
            line += `\\nTraceback:\\n${l.traceback}`;
          }
          return line;
        }
        return String(l);
      }).join('\\n');

      navigator.clipboard.writeText(textToCopy).then(() => {
        showToast("Telemetry logs copied to clipboard!");
      }).catch(err => {
        alert("Could not copy logs: " + err);
      });
    }

    // ==========================================
    // Waypoints Management
    // ==========================================
    let currentApproachWps = [];
    let currentRetreatWps = [];

    async function loadWaypoints() {
      try {
        const res = await fetch('/api/waypoints');
        const data = await res.json();
        currentApproachWps = data.approach || [];
        currentRetreatWps = data.retreat || [];
        const fn = document.getElementById('wp-cfg-filename');
        if (fn && data.waypoints_file) {
          fn.innerText = "config/" + data.waypoints_file;
        }
        renderWaypointsList('approach', currentApproachWps);
        renderWaypointsList('retreat', currentRetreatWps);
        updateWaypointsBadge(currentApproachWps.length, currentRetreatWps.length);
      } catch (err) {
        console.error("Failed to load waypoints", err);
      }
    }

    function updateWaypointsBadge(appCount, retCount) {
      const badge = document.getElementById('badge-waypoints');
      if (badge) {
        const total = (appCount || 0) + (retCount || 0);
        badge.innerText = `${total} WPs (${appCount || 0} App / ${retCount || 0} Ret)`;
      }
    }

    function renderWaypointsList(type, wps) {
      const containerId = type === 'approach' ? 'approach-wps-list' : 'retreat-wps-list';
      const countId = type === 'approach' ? 'approach-count' : 'retreat-count';
      const container = document.getElementById(containerId);
      if (!container) return;
      document.getElementById(countId).innerText = `${wps.length} waypoint${wps.length === 1 ? '' : 's'}`;

      if (wps.length === 0) {
        container.innerHTML = `<span class="empty-queue-msg">No ${type} waypoints configured.</span>`;
        return;
      }

      container.innerHTML = wps.map((wp, idx) => {
        const isJoint = ('joints' in wp && Array.isArray(wp.joints));
        const isAbs = !isJoint && ('x' in wp || 'y' in wp || 'z' in wp);
        let modeBadge = '';
        let posText = '';
        let rpyText = '';

        if (isJoint) {
          modeBadge = `<span class="wp-badge-mode" style="background:rgba(6,182,212,0.15);color:#22d3ee;border:1px solid rgba(6,182,212,0.4);">JOINTS</span>`;
          const jDeg = wp.joints.map(r => (r * 180 / Math.PI).toFixed(1) + '°');
          posText = `J: [${jDeg.join(', ')}]`;
          rpyText = `Rad: [${wp.joints.map(r => Number(r).toFixed(3)).join(', ')}]`;
        } else if (isAbs) {
          modeBadge = `<span class="wp-badge-mode wp-badge-abs">ABS</span>`;
          posText = `X: ${wp.x ?? 0}m, Y: ${wp.y ?? 0}m, Z: ${wp.z ?? 0}m`;
          rpyText = `RPY: [${wp.roll ?? 0}, ${wp.pitch ?? 0}, ${wp.yaw ?? 0}] rad`;
        } else {
          modeBadge = `<span class="wp-badge-mode wp-badge-rel">REL</span>`;
          posText = `dx: ${wp.dx ?? 0}m, dy: ${wp.dy ?? 0}m, dz: ${wp.dz ?? 0}m`;
          rpyText = `RPY: [${wp.roll ?? 0}, ${wp.pitch ?? 0}, ${wp.yaw ?? 0}] rad`;
        }

        const durText = `${wp.duration ?? 2.0}s`;
        const gripText = wp.gripper ? ` | Grip: ${wp.gripper}` : '';
        const pauseText = wp.pause ? ` | Pause` : '';

        return `
          <div class="wp-item-row">
            <div class="wp-item-desc">
              <div><strong>#${idx + 1}</strong> ${modeBadge} <span style="font-family:monospace; color:${isJoint ? '#22d3ee' : '#38bdf8'};">${posText}</span></div>
              <div style="font-size:0.75rem; color:var(--text-muted);">${rpyText} | ${durText}${gripText}${pauseText}</div>
            </div>
            <button class="queue-item-remove" onclick="removeWaypoint('${type}', ${idx})" title="Remove waypoint">X</button>
          </div>
        `;
      }).join('');
    }

    function toggleWpCoordLabels(prefix) {
      const mode = document.getElementById(`${prefix}-mode`).value;
      if (mode === 'relative') {
        document.getElementById(`${prefix}-lbl-x`).innerText = 'dx (m)';
        document.getElementById(`${prefix}-lbl-y`).innerText = 'dy (m)';
        document.getElementById(`${prefix}-lbl-z`).innerText = 'dz (m)';
      } else {
        document.getElementById(`${prefix}-lbl-x`).innerText = 'X (m)';
        document.getElementById(`${prefix}-lbl-y`).innerText = 'Y (m)';
        document.getElementById(`${prefix}-lbl-z`).innerText = 'Z (m)';
      }
    }

    async function addWaypoint(type) {
      const p = type === 'approach' ? 'app' : 'ret';
      const mode = document.getElementById(`${p}-mode`).value;
      const xVal = parseFloat(document.getElementById(`${p}-pos-x`).value) || 0.0;
      const yVal = parseFloat(document.getElementById(`${p}-pos-y`).value) || 0.0;
      const zVal = parseFloat(document.getElementById(`${p}-pos-z`).value) || 0.0;
      const roll = parseFloat(document.getElementById(`${p}-rot-roll`).value) || 0.0;
      const pitch = parseFloat(document.getElementById(`${p}-rot-pitch`).value) || 0.0;
      const yaw = parseFloat(document.getElementById(`${p}-rot-yaw`).value) || 0.0;
      const duration = parseFloat(document.getElementById(`${p}-duration`).value) || 2.0;
      const gripper = document.getElementById(`${p}-gripper`).value;
      const pause = document.getElementById(`${p}-pause`).checked;

      const wp = { duration, roll, pitch, yaw };
      if (mode === 'relative') {
        wp.dx = xVal; wp.dy = yVal; wp.dz = zVal;
      } else {
        wp.x = xVal; wp.y = yVal; wp.z = zVal;
      }
      if (gripper) wp.gripper = gripper;
      if (pause) wp.pause = true;

      if (type === 'approach') {
        currentApproachWps.push(wp);
      } else {
        currentRetreatWps.push(wp);
      }

      await syncWaypoints(true);
    }

    async function removeWaypoint(type, idx) {
      if (type === 'approach') {
        currentApproachWps.splice(idx, 1);
      } else {
        currentRetreatWps.splice(idx, 1);
      }
      await syncWaypoints(true);
    }

    async function syncWaypoints(saveToDisk = true) {
      renderWaypointsList('approach', currentApproachWps);
      renderWaypointsList('retreat', currentRetreatWps);
      updateWaypointsBadge(currentApproachWps.length, currentRetreatWps.length);
      try {
        const res = await fetch('/api/waypoints', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ approach: currentApproachWps, retreat: currentRetreatWps, save_to_disk: saveToDisk })
        });
        const data = await res.json();
        if (saveToDisk && data.success) {
          showToast("Waypoints updated and saved to " + (data.file || "waypoints.yaml") + "!");
        }
      } catch (err) {
        console.error("Failed to sync waypoints", err);
      }
    }

    async function saveWaypointsToDisk() {
      try {
        const res = await fetch('/api/waypoints', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'save_disk' })
        });
        const data = await res.json();
        if (data.success) {
          showToast("Waypoints saved to " + (data.file || "config/waypoints.yaml") + "!");
        } else {
          alert("Error saving: " + data.error);
        }
      } catch (err) {
        alert("Failed to save waypoints: " + err);
      }
    }

    function parseNum(id, fallback) {
      const el = document.getElementById(id);
      if (!el || el.value === "" || el.value === null || el.value === undefined) return fallback;
      const v = parseFloat(el.value);
      return isNaN(v) ? fallback : v;
    }

    let cfgDebounceTimer = null;
    function onConfigInputChange() {
      const statusEl = document.getElementById('cfg-sync-status');
      if (statusEl) {
        statusEl.innerText = "Saving changes to default.yaml...";
        statusEl.style.color = "#f59e0b";
      }
      clearTimeout(cfgDebounceTimer);
      cfgDebounceTimer = setTimeout(() => {
        saveConfiguration(true);
      }, 350);
    }

    function initConfigListeners() {
      document.querySelectorAll('#tab-config input, #tab-config select').forEach(el => {
        el.removeEventListener('change', onConfigInputChange);
        el.removeEventListener('blur', onConfigInputChange);
        el.addEventListener('change', onConfigInputChange);
        el.addEventListener('blur', onConfigInputChange);
      });
    }

    // Load Configuration into form
    async function loadConfiguration() {
      try {
        const res = await fetch('/api/config');
        const cfg = await res.json();
        
        const cam = cfg.camera || {};
        document.getElementById('cfg-cam-id').value = cam.id ?? '0';
        document.getElementById('cfg-cam-focus').value = cam.focus ?? '0';
        document.getElementById('cfg-cam-autofocus').value = cam.autofocus === true ? 'true' : 'false';
        document.getElementById('cfg-cam-intrinsics').value = cam.intrinsics_file ?? '';

        const gp = (cfg.grasp_pipeline || {}).grasp || {};
        document.getElementById('cfg-table-z').value = gp.table_z_m ?? 0.17;
        document.getElementById('cfg-pregrasp-offset').value = gp.pregrasp_offset_m ?? 0.15;
        document.getElementById('cfg-insertion-depth').value = gp.insertion_depth_m ?? 0.0;
        document.getElementById('cfg-grasp-z-offset').value = gp.grasp_z_offset_m ?? -0.02;

        const cal = cfg.calibration || {};
        document.getElementById('cfg-he-file').value = cal.hand_eye_file ?? '';
        const comp = cal.hand_eye_compensation_m || {};
        document.getElementById('cfg-he-comp-x').value = comp.x ?? 0.0;
        document.getElementById('cfg-he-comp-y').value = comp.y ?? 0.0;
        document.getElementById('cfg-he-comp-z').value = comp.z ?? 0.0;

        const rp = (cfg.robot || {}).ready_pose || {};
        document.getElementById('cfg-rp-x').value = rp.x ?? 0.3;
        document.getElementById('cfg-rp-y').value = rp.y ?? 0.0;
        document.getElementById('cfg-rp-z').value = rp.z ?? 0.3;
        document.getElementById('cfg-rp-pitch').value = rp.pitch ?? 0.7;

        const grp = (cfg.robot || {}).gripper || {};
        document.getElementById('cfg-gripper-ready-open').value = grp.ready_open_m ?? 0.055;
        document.getElementById('cfg-gripper-fixed-close').value = grp.fixed_close_m ?? 0.030;
        document.getElementById('cfg-gripper-close-offset').value = grp.close_offset_m ?? 0.010;

        const statusEl = document.getElementById('cfg-sync-status');
        if (statusEl) {
          statusEl.innerText = "Synced with default.yaml";
          statusEl.style.color = "#10b981";
        }
      } catch (err) {
        console.error("Failed to load config", err);
      }
    }

    async function saveConfiguration(silent = false) {
      const statusEl = document.getElementById('cfg-sync-status');
      const payload = {
        camera: {
          id: document.getElementById('cfg-cam-id').value,
          focus: parseNum('cfg-cam-focus', 0),
          autofocus: document.getElementById('cfg-cam-autofocus').value === 'true',
          intrinsics_file: document.getElementById('cfg-cam-intrinsics').value,
        },
        grasp_pipeline: {
          grasp: {
            table_z_m: parseNum('cfg-table-z', 0.17),
            pregrasp_offset_m: parseNum('cfg-pregrasp-offset', 0.15),
            insertion_depth_m: parseNum('cfg-insertion-depth', 0.0),
            grasp_z_offset_m: parseNum('cfg-grasp-z-offset', -0.02),
          }
        },
        calibration: {
          hand_eye_file: document.getElementById('cfg-he-file').value,
          hand_eye_compensation_m: {
            x: parseNum('cfg-he-comp-x', 0.0),
            y: parseNum('cfg-he-comp-y', 0.0),
            z: parseNum('cfg-he-comp-z', 0.0),
          }
        },
        robot: {
          ready_pose: {
            x: parseNum('cfg-rp-x', 0.3),
            y: parseNum('cfg-rp-y', 0.0),
            z: parseNum('cfg-rp-z', 0.3),
            pitch: parseNum('cfg-rp-pitch', 0.7),
          },
          gripper: {
            ready_open_m: parseNum('cfg-gripper-ready-open', 0.055),
            fixed_close_m: document.getElementById('cfg-gripper-fixed-close').value.trim() !== "" ? parseNum('cfg-gripper-fixed-close', null) : null,
            close_offset_m: parseNum('cfg-gripper-close-offset', 0.010),
          }
        }
      };

      try {
        const res = await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
          const now = new Date().toLocaleTimeString();
          if (statusEl) {
            statusEl.innerText = `Synced with default.yaml (${now})`;
            statusEl.style.color = "#10b981";
          }
          if (!silent) {
            showToast("Configuration Saved to default.yaml!");
          }
        } else {
          if (statusEl) {
            statusEl.innerText = `Error saving: ${data.error || 'Server error'}`;
            statusEl.style.color = "#ef4444";
          }
          if (!silent) alert("Error saving: " + data.error);
        }
      } catch (err) {
        if (statusEl) {
          statusEl.innerText = `Failed to save: ${err}`;
          statusEl.style.color = "#ef4444";
        }
        if (!silent) alert("Failed to save config: " + err);
      }
    }

    async function testGripperWidth() {
      const widthVal = parseFloat(document.getElementById('cfg-gripper-ready-open').value);
      if (isNaN(widthVal) || widthVal < 0 || widthVal > 0.09) {
        alert("Please enter a valid gripper width between 0.0 and 0.09 meters (e.g. 0.055).");
        return;
      }
      try {
        const res = await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'test_gripper', width_m: widthVal })
        });
        const data = await res.json();
        if (data.success) {
          showToast(`Gripper test commanded to ${(widthVal * 100).toFixed(1)} cm`);
        } else {
          showToast(data.error || "Failed to command gripper", "error");
        }
      } catch (err) {
        showToast("Error commanding gripper: " + err, "error");
      }
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.innerText = msg;
      t.style.display = 'block';
      setTimeout(() => { t.style.display = 'none'; }, 2500);
    }

    // --------------------------------------------------------------------------
    // TEACH & RECORD MODE LOGIC
    // --------------------------------------------------------------------------
    let currentTeachMode = 'cartesian'; // 'cartesian' or 'joints'
    let teachModeActive = false;

    function setTeachSlideMode(mode) {
      currentTeachMode = mode;
      const btnCart = document.getElementById('btn-mode-cartesian');
      const btnJoints = document.getElementById('btn-mode-joints');
      const hudCart = document.getElementById('teach-hud-cartesian');
      const hudJoints = document.getElementById('teach-hud-joints');
      const modeInd = document.getElementById('teach-mode-indicator');
      const recBtn = document.getElementById('btn-record-pose');

      if (mode === 'cartesian') {
        if (btnCart) { btnCart.style.background = '#2563eb'; btnCart.style.color = '#fff'; }
        if (btnJoints) { btnJoints.style.background = 'transparent'; btnJoints.style.color = '#94a3b8'; }
        if (hudCart) hudCart.style.display = 'block';
        if (hudJoints) hudJoints.style.display = 'none';
        if (modeInd) modeInd.innerText = 'ACTIVE: CARTESIAN (XYZ RPY)';
        if (recBtn) recBtn.innerText = 'Record Current Pose (saves as Cartesian)';
      } else {
        if (btnCart) { btnCart.style.background = 'transparent'; btnCart.style.color = '#94a3b8'; }
        if (btnJoints) { btnJoints.style.background = '#06b6d4'; btnJoints.style.color = '#fff'; }
        if (hudCart) hudCart.style.display = 'none';
        if (hudJoints) hudJoints.style.display = 'block';
        if (modeInd) modeInd.innerText = 'ACTIVE: JOINT ANGLES (J1-J6)';
        if (recBtn) recBtn.innerText = 'Record Current Pose (saves as Joint Angles)';
      }
    }

    async function fetchTeachPose() {
      try {
        const res = await fetch('/api/teach/pose');
        if (!res.ok) return;
        const data = await res.json();
        
        teachModeActive = Boolean(data.active);

        // Update status badge & toggle button
        const badge = document.getElementById('teach-status-badge');
        const toggleBtn = document.getElementById('btn-teach-toggle');
        if (badge) {
          if (teachModeActive) {
            badge.innerText = 'TEACH MODE ACTIVE (FLOATING)';
            badge.style.background = 'rgba(16, 185, 129, 0.2)';
            badge.style.color = '#34d399';
            badge.style.borderColor = '#10b981';
          } else {
            badge.innerText = 'TEACH MODE INACTIVE';
            badge.style.background = 'rgba(148, 163, 184, 0.15)';
            badge.style.color = '#94a3b8';
            badge.style.borderColor = '#475569';
          }
        }

        if (toggleBtn) {
          if (teachModeActive) {
            toggleBtn.innerText = 'Stop Teach Mode & Safe Home';
            toggleBtn.style.background = '#10b981';
            toggleBtn.style.color = '#fff';
          } else {
            toggleBtn.innerText = 'Start Teach Mode (Gravity Compensation)';
            toggleBtn.style.background = '#f59e0b';
            toggleBtn.style.color = '#000';
          }
        }

        // Update Cartesian HUD
        if (data.cartesian) {
          const c = data.cartesian;
          const setT = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
          setT('hud-val-x', (c.x >= 0 ? '+' : '') + Number(c.x).toFixed(4) + ' m');
          setT('hud-val-y', (c.y >= 0 ? '+' : '') + Number(c.y).toFixed(4) + ' m');
          setT('hud-val-z', (c.z >= 0 ? '+' : '') + Number(c.z).toFixed(4) + ' m');
          setT('hud-val-roll', `${Number(c.roll).toFixed(2)} rad (${Number(c.roll_deg).toFixed(1)}°)`);
          setT('hud-val-pitch', `${Number(c.pitch).toFixed(2)} rad (${Number(c.pitch_deg).toFixed(1)}°)`);
          setT('hud-val-yaw', `${Number(c.yaw).toFixed(2)} rad (${Number(c.yaw_deg).toFixed(1)}°)`);
        }

        // Update Joints HUD
        if (data.joints && data.joints_deg) {
          const jDeg = data.joints_deg;
          const jRad = data.joints;
          for (let i = 0; i < 6; i++) {
            const elDeg = document.getElementById(`hud-val-j${i+1}`);
            const elRad = document.getElementById(`hud-val-j${i+1}-rad`);
            if (elDeg) elDeg.innerText = `${Number(jDeg[i]).toFixed(1)}°`;
            if (elRad) elRad.innerText = `${Number(jRad[i]).toFixed(3)} rad`;
          }
        }

        // Update Gripper Display
        const gripEl = document.getElementById('teach-gripper-display');
        if (gripEl && data.gripper_width_cm !== undefined) {
          gripEl.innerText = `Gripper: ${data.gripper_width_cm} cm`;
        }

        // Update Alternate Mode Summary
        const altSum = document.getElementById('teach-hud-alt-summary');
        if (altSum) {
          if (currentTeachMode === 'cartesian') {
            const degs = (data.joints_deg || []).map(d => Number(d).toFixed(1) + '°');
            altSum.innerText = `Joints: [${degs.join(', ')}]`;
          } else if (data.cartesian) {
            const c = data.cartesian;
            altSum.innerText = `Cartesian: X:${c.x.toFixed(3)}m Y:${c.y.toFixed(3)}m Z:${c.z.toFixed(3)}m | Pitch:${c.pitch_deg.toFixed(1)}°`;
          }
        }

        // Render Staged Waypoints Board
        renderTeachStagedWaypoints(data.staged_waypoints || []);
      } catch (err) {
        // silent catch on network glitch
      }
    }

    let lastRenderedStagedJson = null;
    function renderTeachStagedWaypoints(staged, force = false) {
      const container = document.getElementById('teach-staged-container');
      const countBadge = document.getElementById('teach-staged-count');
      if (!container) return;

      const stagedList = staged || [];
      const stagedJson = JSON.stringify(stagedList);
      if (!force && stagedJson === lastRenderedStagedJson) return;
      lastRenderedStagedJson = stagedJson;

      if (countBadge) {
        countBadge.innerText = `${stagedList.length} STAGED`;
      }

      if (stagedList.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 40px 16px; color: var(--text-muted); font-size: 0.85rem; border: 1px dashed var(--card-border); border-radius: 6px;">
            No waypoints recorded yet.<br>Move arm to desired target pose and click "Record Current Pose".
          </div>
        `;
        return;
      }

      container.innerHTML = stagedList.map((wp, idx) => {
        const isJoint = wp.mode === 'joints';
        const badge = isJoint
          ? `<span class="wp-badge-mode" style="background:rgba(6,182,212,0.15);color:#22d3ee;border:1px solid rgba(6,182,212,0.4);">JOINTS</span>`
          : `<span class="wp-badge-mode wp-badge-abs">XYZ RPY</span>`;

        let mainCoord = '';
        let subCoord = '';

        if (isJoint) {
          const jDeg = (wp.joints_deg || []).map(d => Number(d).toFixed(1) + '°');
          mainCoord = `J: [${jDeg.join(', ')}]`;
          subCoord = `Rad: [${(wp.joints || []).map(r => Number(r).toFixed(3)).join(', ')}]`;
        } else {
          mainCoord = `X: ${Number(wp.x).toFixed(3)}m, Y: ${Number(wp.y).toFixed(3)}m, Z: ${Number(wp.z).toFixed(3)}m`;
          const rDeg = (Number(wp.roll) * 180 / Math.PI).toFixed(1);
          const pDeg = (Number(wp.pitch) * 180 / Math.PI).toFixed(1);
          const yDeg = (Number(wp.yaw) * 180 / Math.PI).toFixed(1);
          subCoord = `RPY: [${rDeg}°, ${pDeg}°, ${yDeg}°] (${Number(wp.roll).toFixed(2)}, ${Number(wp.pitch).toFixed(2)}, ${Number(wp.yaw).toFixed(2)} rad)`;
        }

        const dur = wp.duration ?? 2.0;
        const grip = wp.gripper ?? 'keep';

        return `
          <div style="background: #0f131a; border: 1px solid var(--card-border); border-radius: 6px; padding: 10px 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <div style="display: flex; gap: 8px; align-items: center;">
                <strong style="color: #cbd5e1; font-size: 0.9rem;">#${idx + 1}</strong>
                ${badge}
              </div>
              <button class="queue-item-remove" style="padding: 4px 12px; font-size: 0.8rem; font-weight: 700; border-radius: 4px; background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid #ef4444; cursor: pointer;" onclick="discardTeachPose(${idx})" title="Discard this pose">Discard</button>
            </div>
            <div style="font-family: monospace; font-size: 0.85rem; color: ${isJoint ? '#22d3ee' : '#38bdf8'}; font-weight: 600; margin-bottom: 4px;">
              ${mainCoord}
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px;">
              ${subCoord}
            </div>
            <div style="font-size: 0.75rem; color: #64748b; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4px;">
              Duration: <span style="color: #cbd5e1; font-family: monospace;">${dur}s</span> | Gripper: <span style="color: #cbd5e1; font-family: monospace;">${grip}</span>
            </div>
          </div>
        `;
      }).join('');
    }

    async function toggleTeachMode() {
      try {
        const action = teachModeActive ? 'stop' : 'start';
        const res = await fetch('/api/teach/mode', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: action}),
        });
        const data = await res.json();
        if (data.message) showToast(data.message);
        setTimeout(fetchTeachPose, 300);
      } catch (err) {
        showToast('Error toggling teach mode');
      }
    }

    async function recordTeachPose() {
      try {
        const durInput = document.getElementById('teach-record-duration');
        const gripSelect = document.getElementById('teach-record-gripper');
        const duration = parseFloat(durInput ? durInput.value : '2.0') || 2.0;
        const gripper = gripSelect ? gripSelect.value : 'keep';

        const res = await fetch('/api/teach/stage', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            action: 'stage_add',
            mode: currentTeachMode,
            duration: duration,
            gripper: gripper,
          }),
        });
        const data = await res.json();
        if (data.success) {
          showToast(`Recorded Pose #${data.count} as [${currentTeachMode.toUpperCase()}]`);
          renderTeachStagedWaypoints(data.staged_waypoints || [], true);
        }
      } catch (err) {
        showToast('Error recording pose');
      }
    }

    async function discardTeachPose(idx) {
      try {
        const res = await fetch('/api/teach/stage', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: 'stage_discard', index: idx}),
        });
        const data = await res.json();
        if (data.success) {
          showToast(`Discarded pose #${idx + 1}`);
          renderTeachStagedWaypoints(data.staged_waypoints || [], true);
        }
      } catch (err) {
        showToast('Error discarding pose');
      }
    }

    async function clearTeachStaging() {
      if (!confirm('Are you sure you want to discard all staged waypoints?')) return;
      try {
        const res = await fetch('/api/teach/stage', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: 'stage_clear'}),
        });
        const data = await res.json();
        if (data.success) {
          showToast('All staged poses cleared');
          renderTeachStagedWaypoints([], true);
        }
      } catch (err) {
        showToast('Error clearing staging');
      }
    }

    async function commitTeachWaypoints(target) {
      try {
        const res = await fetch('/api/teach/commit', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: 'commit', target: target}),
        });
        const data = await res.json();
        if (data.success) {
          showToast(data.message || `Exported to ${target}!`);
          loadWaypoints();
          if (target === 'ready_pose') {
            loadConfiguration();
          }
        } else {
          showToast(data.error || 'Failed to export');
        }
      } catch (err) {
        showToast('Error committing waypoints');
      }
    }

    // Auto-poll state every 200ms
    setInterval(updateState, 200);
    setInterval(() => {
      const teachTab = document.getElementById('tab-teach');
      if (teachTab && teachTab.classList.contains('active')) {
        fetchTeachPose();
      }
    }, 200);
    updateState();
    loadWaypoints();
    loadConfiguration().then(() => {
      initConfigListeners();
    });

    try {
      const savedTab = sessionStorage.getItem('active_tab');
      if (savedTab && ['station', 'waypoints', 'config', 'teach'].includes(savedTab)) {
        switchTab(savedTab);
      }
    } catch(e) {}
  </script>
</body>
</html>
"""


class MainIndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        state: SystemState = self.application.settings["system_state"]
        state.check_and_reload_if_modified()
        self.set_header("Content-Type", "text/html")
        self.write(INDEX_HTML)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def make_app(state: SystemState, robot_worker: RobotWorker) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", MainIndexHandler),
            (r"/video_feed", VideoStreamHandler),
            (r"/api/status", StatusApiHandler),
            (r"/api/queue", QueueApiHandler),
            (r"/api/action", ActionApiHandler),
            (r"/api/waypoints", WaypointsApiHandler),
            (r"/api/config", ConfigApiHandler),
            (r"/api/teach", TeachApiHandler),
            (r"/api/teach/pose", TeachApiHandler),
            (r"/api/teach/mode", TeachApiHandler),
            (r"/api/teach/stage", TeachApiHandler),
            (r"/api/teach/commit", TeachApiHandler),
        ],
        system_state=state,
        robot_worker=robot_worker,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Web GUI for Fixed-Path Visual Grasping")
    parser.add_argument("--config", default="config/default.yaml", help="Path to default.yaml")
    parser.add_argument("--waypoints-config", default="config/waypoints.yaml", help="Path to waypoints.yaml (default: config/waypoints.yaml)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve the Web UI on (default: 8000)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate motion without commanding hardware")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    wp_cfg_path = PROJECT_ROOT / args.waypoints_config
    state = SystemState(cfg_path, waypoints_cfg_path=wp_cfg_path, dry_run=args.dry_run)

    # Bridge Python logging warnings and errors to GUI telemetry
    log_bridge = WebGuiLoggingHandler(state)
    logging.getLogger().addHandler(log_bridge)

    # Start Vision Worker
    vision_worker = VisionWorker(state)
    vision_worker.start()

    # Start Robot Worker
    robot_worker = RobotWorker(state)
    robot_worker.start()

    # Create & Start Tornado App
    app = make_app(state, robot_worker)
    app.listen(args.port)

    print("\n" + "=" * 60)
    print("  reBot Arm B601 — Web Visual Grasping Station")
    print("=" * 60)
    print(f"  -> Local Web UI:    http://localhost:{args.port}")
    print(f"  -> Config File:     {cfg_path}")
    print(f"  -> Dry Run Mode:    {args.dry_run}")
    print("=" * 60 + "\n")

    shutdown_done = False

    def _shutdown_handler(sig=None, frame=None) -> None:
        nonlocal shutdown_done
        if shutdown_done:
            return
        shutdown_done = True
        print("\n[Shutdown] Terminal interrupt received (Ctrl+C). Initiating clean shutdown...")
        print("[Shutdown] Disabling robot motors and stopping workers...")
        try:
            robot_worker.shutdown()
        except Exception as e:
            print(f"[Shutdown] Error during robot shutdown: {e}")
        vision_worker.running = False
        try:
            tornado.ioloop.IOLoop.current().add_callback_from_signal(tornado.ioloop.IOLoop.current().stop)
        except Exception:
            pass
        vision_worker.join(timeout=1.5)
        if robot_worker.is_alive():
            robot_worker.join(timeout=1.5)
        print("[Shutdown] Robot confirmed disabled. Server shutdown complete.\n")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    atexit.register(_shutdown_handler)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        _shutdown_handler()
    finally:
        _shutdown_handler()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
