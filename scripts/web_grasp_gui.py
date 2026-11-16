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
import collections
import copy
import json
import logging
import os
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
except ImportError:
    RebotArm = None  # type: ignore
    RebotArmEndPose = None  # type: ignore

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


def _wait_motion(controller: Any, duration: float, settle_time: float = 0.4) -> None:
    if controller is None:
        time.sleep(duration)
        return
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 2.0)
    else:
        time.sleep(duration)
    if settle_time > 0:
        time.sleep(settle_time)


def _move_ready(
    controller: Any,
    ready_cfg: dict[str, Any],
    grasp_driver: Optional[GraspDriver] = None,
    open_at_ready: bool = True,
    dry_run: bool = False,
    ready_open_m: Optional[float] = None,
) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    x = float(ready_cfg.get("x", 0.35))
    y = float(ready_cfg.get("y", 0.0))
    z = float(ready_cfg.get("z", 0.28))
    roll = float(ready_cfg.get("roll", 0.0))
    pitch = float(ready_cfg.get("pitch", 0.6))
    yaw = float(ready_cfg.get("yaw", 0.0))
    if not dry_run and controller is not None:
        controller.move_to_traj(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration)
        _wait_motion(controller, duration)
        if open_at_ready and grasp_driver is not None:
            open_width = ready_open_m if ready_open_m is not None else GRIPPER_READY_OPEN_M
            grasp_driver.open_gripper(distance_m=open_width)
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
    duration = float(wp.get("duration", 2.0))
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

        # Activity log & Error telemetry
        self.logs: collections.deque[LogEntry] = collections.deque(maxlen=500)
        self.last_error: Optional[dict[str, Any]] = None
        self.error_count: int = 0

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
            self.log("Configuration and waypoints reloaded from disk.")

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
        self.command_queue: collections.deque[tuple[str, Any]] = collections.deque()
        self.command_event = threading.Event()

    def submit(self, action: str, data: Any = None) -> None:
        self.command_queue.append((action, data))
        self.command_event.set()

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
                if action == "grasp_queue":
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
                elif action == "resume":
                    self.state.pause_event.set()
                elif action == "unfreeze":
                    with self.state.lock:
                        self.state.status = "live"
                        self.state.status_detail = "Live Camera Preview"
            except Exception as e:
                self.state.log_error(f"Robot action '{action}' error: {e}", exc=e, source="RobotWorker")
                with self.state.lock:
                    self.state.status = "live"
                    self.state.status_detail = f"Error during {action}"

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
        self.state.log(f"Starting sequential execution of {total} queued tube(s)...")

        for idx, item in enumerate(items_to_pick, start=1):
            label = item["label"]
            p_base = item["p_base"]
            jaw_width = item["jaw_width_m"]
            with self.state.lock:
                self.state.status_detail = f"Grasping {label} ({idx}/{total})"
            self.state.log(f"--- Picking {label} ({idx}/{total}): base=({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f}) ---")

            xg, yg, zg = float(p_base[0]), float(p_base[1]), float(p_base[2])
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
                    if wp.get("pause", False):
                        prompt = wp.get("prompt", f"Paused at Approach WP #{aw_idx}")
                        with self.state.lock:
                            self.state.status = "paused"
                            self.state.pause_prompt = prompt
                            self.state.pause_event.clear()
                        self.state.log(f"  ⏸ {prompt} -> Waiting for web 'Resume' click...")
                        self.state.pause_event.wait()
                        with self.state.lock:
                            self.state.status = "executing"
                            self.state.status_detail = f"Grasping {label} ({idx}/{total})"
                        self.state.log("  ▶ Resumed motion.")

                    wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, rpy_cfg)
                    self.state.log(f"  Approach WP #{aw_idx}: ({wx:.3f}, {wy:.3f}, {wz:.3f})")
                    if not dry:
                        ctrl.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur)
                        _wait_motion(ctrl, wdur)
                        _handle_waypoint_gripper(driver, wp.get("gripper"), dry_run=dry, ready_open_m=self.state.gripper_ready_open_m)
                    else:
                        time.sleep(0.2)

                # 2. Pre-Grasp Hover
                if pg_cfg.get("enabled", True):
                    pre_dur = float(pg_cfg.get("duration", 5.0))
                    self.state.log(f"  Move to Pregrasp: ({xp:.3f}, {yp:.3f}, {zp:.3f})")
                    if not dry:
                        ctrl.move_to_traj(xp, yp, zp, rxg, ryg, rzg, duration=pre_dur)
                        _wait_motion(ctrl, pre_dur)
                    else:
                        time.sleep(0.3)

                # 3. Move to Grasp Target
                self.state.log(f"  Move to Grasp: ({xg:.3f}, {yg:.3f}, {zg:.3f})")
                if not dry:
                    ctrl.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=GRASP_MOVE_DURATION)
                    _wait_motion(ctrl, GRASP_MOVE_DURATION)
                else:
                    time.sleep(0.3)

                # 4. Close Gripper
                fixed_close = self.state.gripper_fixed_close_m
                close_offset = self.state.gripper_close_offset_m
                target_jaw = max(0.0, float(fixed_close)) if fixed_close is not None else max(0.0, jaw_width - close_offset)
                self.state.log(f"  Closing gripper on {label} (target: {target_jaw*100:.1f}cm)...")
                if not dry and driver is not None:
                    driver.grasp(force=1.5)
                else:
                    time.sleep(0.2)

                # 5. Strict Retreat Waypoints (as configured, no extra open_gripper)
                self.state.log("  Executing Retreat Waypoints...")
                for rw_idx, wp in enumerate(retreat_wps, start=1):
                    # Check for pause waypoint
                    if wp.get("pause", False):
                        prompt = wp.get("prompt", f"Paused at Retreat WP #{rw_idx}")
                        with self.state.lock:
                            self.state.status = "paused"
                            self.state.pause_prompt = prompt
                            self.state.pause_event.clear()
                        self.state.log(f"  ⏸ {prompt} -> Waiting for web 'Resume' click...")
                        self.state.pause_event.wait()
                        with self.state.lock:
                            self.state.status = "executing"
                            self.state.status_detail = f"Grasping {label} ({idx}/{total})"
                        self.state.log("  ▶ Resumed motion.")

                    wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, rpy_cfg)
                    self.state.log(f"  Retreat WP #{rw_idx}: ({wx:.3f}, {wy:.3f}, {wz:.3f})")
                    if not dry:
                        ctrl.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur)
                        _wait_motion(ctrl, wdur)
                        _handle_waypoint_gripper(driver, wp.get("gripper"), dry_run=dry, ready_open_m=self.state.gripper_ready_open_m)
                    else:
                        time.sleep(0.3)

                # 6. Return to Ready Pose (Approach Waypoint 0) with open_at_ready=True
                self.state.log("  Returning to Ready Pose...")
                _move_ready(ctrl, ready_cfg, grasp_driver=driver, open_at_ready=True, dry_run=dry, ready_open_m=self.state.gripper_ready_open_m)

                # Remove item from state queue
                with self.state.lock:
                    self.state.queue = [q for q in self.state.queue if q["queue_uid"] != item["queue_uid"]]

            except Exception as e:
                self.state.log_error(f"Grasp execution aborted on {label}: {e}", exc=e, source="RobotWorker")
                break

        with self.state.lock:
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"
        if not self.state.queue:
            self.state.log_success("All queued grasps completed successfully!")

    def _execute_macro_center(self, tube_id: Optional[int]) -> None:
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
                _wait_motion(ctrl, 3.0)
            else:
                time.sleep(0.5)
            self.state.log_success("Macro Centering complete.")
        except Exception as e:
            self.state.log_error(f"Macro Centering failed: {e}", exc=e, source="RobotWorker")

    def _execute_safe_home(self) -> None:
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
            time.sleep(1.0)
            self.state.log_success("Safe Home complete (simulated).")

        with self.state.lock:
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"

    def _execute_ready_pose(self) -> None:
        with self.state.lock:
            if not self.state.robot_enabled:
                self.state.log_warning("Robot disabled. Enable robot first.", source="RobotWorker")
                return
            self.state.status = "executing"
            self.state.status_detail = "Moving to Ready Pose"
        self.state.log("Moving arm to Ready Pose...")

        ctrl = self.state.controller
        driver = self.state.grasp_driver
        dry = self.state.dry_run or ctrl is None
        ready_cfg = self.state.cfg.get("robot", {}).get("ready_pose", LOCAL_READY_POSE)

        try:
            _move_ready(ctrl, ready_cfg, grasp_driver=driver, open_at_ready=True, dry_run=dry, ready_open_m=self.state.gripper_ready_open_m)
            self.state.log_success("Arm returned to Ready Pose.")
        except Exception as e:
            self.state.log_error(f"Ready pose motion failed: {e}", exc=e, source="RobotWorker")

        with self.state.lock:
            self.state.status = "live"
            self.state.status_detail = "Live Camera Preview"

    def _execute_test_gripper(self, target_width: Any) -> None:
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
        with state.lock:
            with open(state.cfg_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
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

                state.reload_config()
                state.save_waypoints_to_disk()
                self.write(json.dumps({"success": True, "message": "Configuration saved successfully."}))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


# ==============================================================================
# HTML FRONTEND DASHBOARD
# ==============================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>reBot Arm B601 — Pick & Place Web Station</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0f131a;
      --card-bg: #181e2b;
      --card-border: #263143;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --green: #10b981;
      --green-hover: #059669;
      --amber: #f59e0b;
      --amber-hover: #d97706;
      --red: #ef4444;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    
    /* Top Navbar */
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 1.1rem; }
    .nav-tabs { display: flex; gap: 8px; }
    .tab-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: #222b3d;
      color: #fff;
      border-color: var(--card-border);
    }
    .tab-btn:hover:not(.active) { color: var(--text); }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .status-live { background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid var(--green); }
    .status-frozen { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid #60a5fa; }
    .status-executing { background: rgba(245, 158, 11, 0.15); color: var(--amber); border: 1px solid var(--amber); }
    .status-paused { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* Main Container */
    main { flex: 1; padding: 20px; max-width: 1500px; margin: 0 auto; width: 100%; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Layout for Station */
    .station-grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; }
    @media (max-width: 1024px) { .station-grid { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Video Frame */
    .video-container {
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      aspect-ratio: 16 / 9;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .video-container img { width: 100%; height: 100%; object-fit: contain; }

    /* Tubes Button Bar */
    .tube-buttons-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; min-height: 42px; align-items: center; }
    .tube-btn {
      background: #2563eb;
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
      font-size: 0.95rem;
      transition: transform 0.1s, background 0.2s;
    }
    .tube-btn:hover { background: #1d4ed8; transform: translateY(-2px); }
    .tube-btn:active { transform: translateY(0); }
    .tube-btn.queued { background: #059669; }

    /* Queue List */
    .queue-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 200px;
      overflow-y: auto;
      margin-bottom: 16px;
    }
    .queue-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #111622;
      border: 1px solid var(--card-border);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    .queue-item-info { font-weight: 600; }
    .queue-item-coord { color: var(--text-muted); font-size: 0.8rem; margin-left: 8px; }
    .queue-item-remove {
      background: transparent;
      color: var(--red);
      border: none;
      cursor: pointer;
      font-size: 1.1rem;
      padding: 2px 6px;
    }
    .empty-queue-msg { color: var(--text-muted); font-size: 0.85rem; font-style: italic; }

    /* Actions Panel */
    .actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }
    .btn-main {
      padding: 12px;
      border-radius: 8px;
      border: none;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .btn-grasp { background: var(--green); color: #fff; grid-column: span 2; font-size: 1.1rem; }
    .btn-grasp:hover { background: var(--green-hover); }
    .btn-grasp:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-enable { background: #059669; color: #fff; }
    .btn-enable:hover { background: #047857; }
    .btn-disable { background: #dc2626; color: #fff; }
    .btn-disable:hover { background: #b91c1c; }
    .btn-center { background: var(--accent); color: #fff; }
    .btn-center:hover { background: var(--accent-hover); }
    .btn-home { background: #6366f1; color: #fff; }
    .btn-home:hover { background: #4f46e5; }
    .btn-ready { background: #0891b2; color: #fff; }
    .btn-ready:hover { background: #0e7490; }
    .btn-reset { background: #334155; color: #fff; }
    .btn-reset:hover { background: #475569; }
    .btn-resume { background: var(--amber); color: #000; grid-column: span 2; display: none; }
    .btn-resume:hover { background: var(--amber-hover); }
    .btn-resume.active { display: flex; animation: pulse 1s infinite; }

    /* Waypoint Editor Tab Styles */
    .wp-editor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .wp-editor-grid { grid-template-columns: 1fr; } }
    .wps-table-container {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 240px;
      overflow-y: auto;
      margin-bottom: 16px;
      background: #090c12;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px;
    }
    .wp-item-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #141b27;
      padding: 8px 10px;
      border-radius: 4px;
      font-size: 0.85rem;
      border: 1px solid #1f293d;
    }
    .wp-item-desc { display: flex; flex-direction: column; gap: 3px; }
    .wp-badge-mode {
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 700;
      text-transform: uppercase;
      display: inline-block;
      width: fit-content;
    }
    .wp-badge-rel { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .wp-badge-abs { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .wp-form-box {
      background: #111622;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 12px;
    }
    .wp-form-title { font-size: 0.85rem; font-weight: 700; color: var(--text-muted); margin-bottom: 10px; }
    .btn-sm { padding: 4px 10px; border-radius: 4px; border: none; font-size: 0.8rem; font-weight: 600; cursor: pointer; }

    /* Enhanced Console & Debugging Log */
    .console-box {
      background: #060910;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 14px;
      font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
      font-size: 0.86rem;
      line-height: 1.55;
      color: #94a3b8;
      height: 340px;
      min-height: 220px;
      max-height: 600px;
      overflow-y: auto;
      resize: vertical;
      box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.4);
    }
    .console-box::-webkit-scrollbar {
      width: 8px;
    }
    .console-box::-webkit-scrollbar-thumb {
      background: #334155;
      border-radius: 4px;
    }
    .console-box::-webkit-scrollbar-thumb:hover {
      background: #475569;
    }

    /* Log Entry Types */
    .log-line {
      display: flex;
      flex-direction: column;
      padding: 3px 6px;
      margin-bottom: 2px;
      border-radius: 4px;
      border-left: 3px solid transparent;
      word-break: break-word;
      transition: background 0.15s;
    }
    .log-line:hover {
      background: rgba(255, 255, 255, 0.03);
    }
    .log-row-main {
      display: flex;
      gap: 8px;
      align-items: baseline;
    }
    .log-time {
      color: #64748b;
      font-size: 0.78rem;
      flex-shrink: 0;
      font-weight: 500;
    }
    .log-badge {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 3px;
      text-transform: uppercase;
      flex-shrink: 0;
    }
    .log-msg {
      flex: 1;
    }

    /* Info Level */
    .log-info {
      border-left-color: #38bdf8;
      color: #cbd5e1;
    }
    .log-info .log-badge {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
    }

    /* Success Level */
    .log-success {
      border-left-color: #10b981;
      background: rgba(16, 185, 129, 0.06);
      color: #6ee7b7;
    }
    .log-success .log-badge {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
    }

    /* Warning Level */
    .log-warning {
      border-left-color: #f59e0b;
      background: rgba(245, 158, 11, 0.08);
      color: #fde68a;
    }
    .log-warning .log-badge {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
    }

    /* Error Level */
    .log-error {
      border-left-color: #ef4444;
      background: rgba(239, 68, 68, 0.12);
      color: #fca5a5;
      font-weight: 600;
    }
    .log-error .log-badge {
      background: rgba(239, 68, 68, 0.25);
      color: #f87171;
    }

    /* Traceback inside log */
    .log-trace-details {
      margin-top: 4px;
      margin-left: 12px;
      font-size: 0.8rem;
    }
    .log-trace-details summary {
      color: #f87171;
      cursor: pointer;
      font-weight: 500;
      text-decoration: underline;
      user-select: none;
    }
    .log-trace-details summary:hover {
      color: #fca5a5;
    }
    .log-trace-pre {
      background: #020408;
      border: 1px solid #7f1d1d;
      border-radius: 4px;
      padding: 8px 10px;
      margin-top: 4px;
      color: #fca5a5;
      font-size: 0.78rem;
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 200px;
    }

    /* Active Error Banner */
    .active-error-banner {
      background: linear-gradient(135deg, rgba(153, 27, 27, 0.35), rgba(69, 10, 10, 0.45));
      border: 1px solid #dc2626;
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 12px;
      animation: alertPulse 2s infinite ease-in-out;
    }
    @keyframes alertPulse {
      0%, 100% { border-color: #dc2626; box-shadow: 0 0 10px rgba(220, 38, 38, 0.2); }
      50% { border-color: #ef4444; box-shadow: 0 0 16px rgba(239, 68, 68, 0.4); }
    }
    .error-banner-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }
    .error-banner-icon { font-size: 1.1rem; }
    .error-banner-title {
      font-weight: 700;
      font-size: 0.85rem;
      color: #f87171;
      letter-spacing: 0.05em;
    }
    .error-banner-time {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-left: auto;
    }
    .error-banner-dismiss {
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 1.1rem;
      cursor: pointer;
      padding: 0 4px;
      line-height: 1;
    }
    .error-banner-dismiss:hover { color: #fff; }
    .error-banner-msg {
      color: #fecaca;
      font-size: 0.88rem;
      font-weight: 600;
      font-family: ui-monospace, monospace;
      margin-left: 28px;
      word-break: break-word;
    }
    .error-trace-details {
      margin-top: 8px;
      margin-left: 28px;
    }
    .error-trace-details summary {
      color: #f87171;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }
    .error-trace-pre {
      background: #020408;
      border: 1px solid rgba(220, 38, 38, 0.4);
      border-radius: 6px;
      padding: 10px;
      margin-top: 6px;
      color: #fca5a5;
      font-size: 0.78rem;
      font-family: ui-monospace, monospace;
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 250px;
    }

    /* Error count pill */
    .log-pill {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 12px;
      display: inline-block;
    }
    .log-pill-clean {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .log-pill-alert {
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
      border: 1px solid #ef4444;
      animation: pulse 1.5s infinite;
    }

    /* Filter buttons */
    .log-filter-group {
      display: flex;
      background: #090c12;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      overflow: hidden;
    }
    .btn-filter {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 8px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-filter.active {
      background: var(--card-border);
      color: #fff;
    }
    .btn-filter:hover:not(.active) {
      color: #cbd5e1;
    }

    /* Configuration Tab */
    .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .form-group { margin-bottom: 12px; }
    .form-group label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--text-muted); margin-bottom: 4px; }
    .form-group input, .form-group select {
      width: 100%;
      background: #090c12;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 8px;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    .btn-save-cfg {
      background: var(--green);
      color: #fff;
      border: none;
      padding: 12px 24px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      margin-top: 10px;
    }
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--green);
      color: #fff;
      padding: 12px 24px;
      border-radius: 6px;
      font-weight: 600;
      display: none;
      z-index: 100;
    }

    /* Guide Tab */
    .guide-banner {
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%);
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }
    .guide-step-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .guide-step-card {
      background: #151c28;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 18px;
    }
    .guide-step-num {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      background: #2563eb;
      color: #fff;
      font-weight: 800;
      border-radius: 50%;
      font-size: 0.9rem;
      margin-right: 8px;
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
      font-size: 0.88rem;
    }
    .guide-table th {
      background: #111827;
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
      background: #151c28;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px 18px;
    }

    /* Robot Hardware LED Indicator */
    .robot-led-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 5px 12px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--card-border);
      transition: all 0.3s ease;
    }
    .led-bulb {
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 50%;
      transition: all 0.25s ease;
      flex-shrink: 0;
    }
    /* LED ON: Bright glowing emerald green */
    .led-bulb.led-on {
      background-color: #10b981;
      box-shadow: 0 0 6px #10b981, 0 0 14px rgba(16, 185, 129, 0.7);
      border: 1px solid #34d399;
    }
    /* LED OFF: Dim slate gray */
    .led-bulb.led-off {
      background-color: #475569;
      box-shadow: none;
      border: 1px solid #334155;
    }
    /* LED PENDING: Pulsing amber while waiting for feedback */
    .led-bulb.led-pending {
      background-color: #f59e0b;
      box-shadow: 0 0 6px #f59e0b, 0 0 12px rgba(245, 158, 11, 0.6);
      border: 1px solid #fbbf24;
      animation: led-pulse 0.8s infinite alternate;
    }
    @keyframes led-pulse {
      0% { opacity: 0.35; }
      100% { opacity: 1; }
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span>🦾</span>
      <span>reBot Arm B601</span>
      <span style="font-weight: 400; color: var(--text-muted);">|</span>
    </div>

    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('station')">🎮 Visual Grasping</button>
      <button class="tab-btn" onclick="switchTab('waypoints')">📍 Waypoints Editor</button>
      <button class="tab-btn" onclick="switchTab('config')">⚙️ Robot & Camera Config</button>
      <button class="tab-btn" onclick="switchTab('guide')">📖 User Guide</button>
    </div>

    <div style="display: flex; gap: 10px; align-items: center;">
      <span id="badge-status" class="status-badge status-live">🟢 LIVE</span>
      <div id="badge-enabled-container" class="robot-led-badge" title="Hardware Motor Driver Feedback">
        <span class="led-bulb led-off" id="led-indicator-bulb"></span>
        <span id="badge-enabled-text" style="color: #94a3b8;">ROBOT DISABLED</span>
      </div>
      <span id="badge-waypoints" class="status-badge" style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid #818cf8; cursor: pointer;" title="Active Robot Waypoints (Click to view & edit)" onclick="switchTab('waypoints')">📍 2 WPs (0 App / 2 Ret)</span>
      <button class="tab-btn" style="border: 1px solid #3b82f6; color: #60a5fa; background: rgba(59, 130, 246, 0.12); padding: 4px 12px; font-size: 0.8rem; font-weight: 700; border-radius: 999px; cursor: pointer;" onclick="switchTab('guide')" title="Open Quick User Guide">❓ Quick Guide</button>
      <span id="badge-mode" style="font-size: 0.8rem; color: var(--text-muted);">Connected</span>
    </div>
  </header>

  <main>
    <!-- TAB 1: VISUAL GRASPING STATION -->
    <div id="tab-station" class="tab-content active">
      <div class="station-grid">
        
        <!-- Left: Live Video Feed -->
        <div>
          <div class="card" style="padding: 10px;">
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
              <span id="tubes-count" style="font-size: 0.8rem; color: var(--text-muted);">0 tubes</span>
            </div>
            <div id="tubes-bar" class="tube-buttons-bar">
              <span class="empty-queue-msg">Scanning table...</span>
            </div>
          </div>

          <!-- Picking Queue -->
          <div class="card">
            <div class="card-title">
              <span>PICKING QUEUE (CUSTOM ORDER)</span>
              <button onclick="clearQueue()" style="background:transparent; border:none; color:var(--red); font-size:0.8rem; cursor:pointer;">Clear All</button>
            </div>
            <div id="queue-box" class="queue-list">
              <span class="empty-queue-msg">No tubes queued. Click buttons above to add.</span>
            </div>

            <!-- Hardware Motor Driver Feedback Indicator Card -->
            <div class="card" style="padding: 10px 14px; margin-bottom: 12px; background: #0c1017; border-left: 4px solid #475569;" id="card-led-border">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                  <span class="led-bulb led-off" id="card-led-bulb" style="width: 14px; height: 14px;"></span>
                  <div>
                    <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Motor Driver Feedback</div>
                    <div id="card-led-text" style="font-size: 0.92rem; font-weight: 700; color: #94a3b8;">Motors Disabled (Torque Off)</div>
                  </div>
                </div>
                <div id="card-led-subtext" style="font-size: 0.75rem; color: var(--text-muted); background: rgba(255,255,255,0.04); padding: 3px 8px; border-radius: 4px;">Hardware feedback offline</div>
              </div>
            </div>

            <!-- Action Buttons -->
            <div class="actions-grid">
              <button id="btn-resume" class="btn-main btn-resume" onclick="resumeMotion()">
                <span>⏸</span> RESUME FROM PAUSE
              </button>
              <button id="btn-grasp" class="btn-main btn-grasp" onclick="graspQueue()">
                <span>▶</span> GRASP QUEUE
              </button>
              <button id="btn-enable" class="btn-main btn-enable" onclick="enableRobot()">
                <span>⚡</span> Enable Robot
              </button>
              <button id="btn-disable" class="btn-main btn-disable" onclick="disableRobot()">
                <span>🛑</span> Disable Robot
              </button>
              <button class="btn-main btn-center" onclick="macroCenter()">
                <span>🎯</span> Macro Center
              </button>
              <button class="btn-main btn-home" onclick="safeHome()">
                <span>🏠</span> Safe Home
              </button>
              <button class="btn-main btn-ready" onclick="readyPose()">
                <span>🦾</span> Ready Pose
              </button>
              <button class="btn-main btn-reset" onclick="unfreezeStream()">
                <span>🔄</span> Reset / Unfreeze
              </button>
            </div>
          </div>

          <!-- System Telemetry & Error Log -->
          <div class="card">
            <div class="card-title" style="flex-wrap: wrap; gap: 8px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span>📟 SYSTEM TELEMETRY &amp; DEBUG LOG</span>
                <span id="log-error-pill" class="log-pill log-pill-clean">0 Errors</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <div class="log-filter-group">
                  <button id="filter-all" class="btn-filter active" onclick="setLogFilter('all')">All</button>
                  <button id="filter-errors" class="btn-filter" onclick="setLogFilter('errors')">⚠️ Errors (<span id="filter-err-count">0</span>)</button>
                </div>
                <label style="display:flex; align-items:center; gap:5px; font-size:0.75rem; color:var(--text-muted); cursor:pointer; user-select:none;">
                  <input type="checkbox" id="chk-autoscroll" checked style="width:auto; cursor:pointer;"> Auto-scroll
                </label>
                <button class="btn-sm" style="background:#334155; color:#fff;" onclick="copyAllLogs()" title="Copy all logs to clipboard">📋 Copy</button>
                <button class="btn-sm" style="background:#1e293b; color:#cbd5e1; border:1px solid #334155;" onclick="clearLogs()" title="Clear logs">🗑️ Clear</button>
                <button class="btn-sm" style="background:#451a03; color:#f59e0b; border:1px solid #78350f;" onclick="triggerTestError()" title="Simulate an error for debugging test">🧪 Test Error</button>
              </div>
            </div>

            <!-- Active Error Banner (Prominent Alert for Debugging) -->
            <div id="active-error-banner" class="active-error-banner" style="display: none;">
              <div class="error-banner-header">
                <span class="error-banner-icon">🚨</span>
                <span class="error-banner-title">RUNTIME ERROR DETECTED</span>
                <span id="error-banner-time" class="error-banner-time"></span>
                <button class="error-banner-dismiss" onclick="dismissActiveError()" title="Dismiss banner">✕</button>
              </div>
              <div id="error-banner-msg" class="error-banner-msg"></div>
              <details id="error-banner-trace-details" class="error-trace-details" style="display: none;">
                <summary>🔍 View Stack Traceback</summary>
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
            <button class="btn-sm" style="background:#334155; color:#fff;" onclick="loadWaypoints()">🔄 Reload</button>
            <button class="btn-sm" style="background:var(--green); color:#fff;" onclick="saveWaypointsToDisk()">💾 Save to waypoints.yaml</button>
          </div>
        </div>
        <div style="background: #111622; border: 1px solid var(--card-border); border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div>
            <span style="font-weight: 700; color: #818cf8;">📌 Active Robot Configuration:</span>
            <span style="font-family: monospace; color: #38bdf8; margin-left: 6px;" id="wp-cfg-filename">config/waypoints.yaml</span>
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">
            Waypoints shown below are the exact active via-points executed by the robot. Changes are saved and persist automatically.
          </div>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 16px;">
          Configure intermediate waypoints before grasping (Approach) and after grasping (Retreat). Position values can be Relative to Grasp Point or Absolute in Base Frame.
        </p>

        <!-- TWO COLUMN GRID: LEFT = APPROACH, RIGHT = RETREAT -->
        <div class="wp-editor-grid">
          
          <!-- Column 1: Approach Waypoints -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title" style="color: #60a5fa;">
              <span>🛫 APPROACH WAYPOINTS (READY POSE → TARGET)</span>
              <span id="approach-count" style="font-size: 0.8rem; color: var(--text-muted);">0 waypoints</span>
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

              <button class="btn-main" style="background:#2563eb; color:#fff; width:100%; padding:10px;" onclick="addWaypoint('approach')">
                + Append Approach Waypoint
              </button>
            </div>
          </div>

          <!-- Column 2: Retreat Waypoints -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title" style="color: #34d399;">
              <span>🛬 RETREAT WAYPOINTS (GRASPED → READY POSE)</span>
              <span id="retreat-count" style="font-size: 0.8rem; color: var(--text-muted);">2 waypoints</span>
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

              <button class="btn-main" style="background:#059669; color:#fff; width:100%; padding:10px;" onclick="addWaypoint('retreat')">
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
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 16px;">
          Adjust camera, table plane, hand-eye trims, and ready pose directly. Changes write directly to disk.
        </p>

        <div class="config-grid">
          <!-- Camera Settings -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">📷 CAMERA & FOCUS</div>
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
            <div class="card-title">📐 GRASP PLANE & OFFSETS</div>
            <div class="form-group">
              <label>Table Z Plane (meters, base frame)</label>
              <input type="number" step="0.005" id="cfg-table-z">
            </div>
            <div class="form-group">
              <label>Pre-Grasp Offset (meters)</label>
              <input type="number" step="0.005" id="cfg-pregrasp-offset">
            </div>
            <div class="form-group">
              <label>Insertion Depth (meters)</label>
              <input type="number" step="0.005" id="cfg-insertion-depth">
            </div>
          </div>

          <!-- Hand-Eye Calibration -->
          <div class="card" style="margin-bottom: 0;">
            <div class="card-title">🔄 HAND-EYE CALIBRATION</div>
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
            <div class="card-title">🦾 READY POSE</div>
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
            <div class="card-title">🤏 GRIPPER CONFIGURATION</div>
            <div class="form-group">
              <label>Ready Open Width (meters, max stroke 0.09m)</label>
              <input type="number" step="0.005" min="0.0" max="0.09" id="cfg-gripper-ready-open" placeholder="0.055">
              <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 3px;">
                Width gripper opens to at Ready Pose and approach (0.055m = 5.5cm).
              </div>
            </div>
            <div class="form-group">
              <label>Fixed Close Width (meters, optional)</label>
              <input type="number" step="0.005" min="0.0" max="0.09" id="cfg-gripper-fixed-close" placeholder="0.030">
              <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 3px;">
                Fixed close target width (0.03m = 3.0cm). Leave blank to use adaptive YOLO detection width.
              </div>
            </div>
            <div class="form-group">
              <label>Close Offset Margin (meters)</label>
              <input type="number" step="0.002" min="0.0" max="0.05" id="cfg-gripper-close-offset" placeholder="0.010">
              <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 3px;">
                Subtracted from YOLO jaw width when closing adaptively (0.01m = 1.0cm).
              </div>
            </div>
            <button class="btn btn-outline" style="width: 100%; margin-top: 6px; font-size: 0.82rem; padding: 7px 12px; border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8;" onclick="testGripperWidth()">
              🤏 Test Gripper Width Now
            </button>
          </div>
        </div>

        <button class="btn-save-cfg" onclick="saveConfiguration()">💾 Save Configuration to default.yaml</button>
      </div>
    </div>

    <!-- TAB 4: USER GUIDE & WALKTHROUGH -->
    <div id="tab-guide" class="tab-content">
      <div class="guide-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
          <div>
            <h2 style="font-size: 1.3rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px;">
              <span>📖</span> Station Operator &amp; Quick-Start Guide
            </h2>
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 6px;">
              A simple, step-by-step walkthrough for lab operators and engineers to pick test tubes with the reBot Arm B601.
            </p>
          </div>
          <button onclick="switchTab('station')" class="btn-main" style="background: #2563eb; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">
            🚀 Open Grasping Station
          </button>
        </div>
      </div>

      <!-- Quick 3-Step Walkthrough -->
      <div class="card">
        <div class="card-title">
          <span>🚀 QUICK-START: HOW TO PICK TUBES IN 3 STEPS</span>
        </div>
        <div class="guide-step-grid">
          
          <div class="guide-step-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
              <span class="guide-step-num">1</span>
              <strong style="color: #60a5fa; font-size: 1rem;">Move Arm to Ready Pose 🦾</strong>
            </div>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
              When the station starts, the robot holds still for safety. Click the blue <strong>"Ready Pose"</strong> button. The robotic arm smoothly moves above the rack, opens its gripper, and aims the camera directly at the test tubes.
            </p>
            <div style="margin-top: 12px; background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; padding: 6px 10px; font-size: 0.8rem; color: #93c5fd;">
              💡 <em>Tip: Always ensure the workspace is clear before moving to Ready Pose.</em>
            </div>
          </div>

          <div class="guide-step-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
              <span class="guide-step-num">2</span>
              <strong style="color: #34d399; font-size: 1rem;">Select Tubes to Pick 🎯</strong>
            </div>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
              Look at the live video stream. Tubes are automatically numbered from left to right (<strong>#1, #2, #3...</strong>). Click the tube buttons in the order you want them picked (e.g. click <strong>#3</strong>, then <strong>#1</strong>). They appear in the <strong>PICKING QUEUE</strong> list.
            </p>
            <div style="margin-top: 12px; background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 6px 10px; font-size: 0.8rem; color: #6ee7b7;">
              💡 <em>Tip: Click ✕ on any tube in the queue if you made a mistake, or click "Clear All".</em>
            </div>
          </div>

          <div class="guide-step-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
              <span class="guide-step-num">3</span>
              <strong style="color: #f59e0b; font-size: 1rem;">Click "GRASP QUEUE" ▶</strong>
            </div>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
              Click the green <strong>"GRASP QUEUE"</strong> button. The camera freezes to lock coordinates, and the arm automatically hovers (+10cm), descends, clamps the tube, lifts straight up, and tilts away safely. Once finished, it returns to live mode!
            </p>
            <div style="margin-top: 12px; background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 6px 10px; font-size: 0.8rem; color: #fde68a;">
              💡 <em>Tip: Never reach into the robot workspace while the status shows "EXECUTING".</em>
            </div>
          </div>

        </div>
      </div>

      <!-- Button Controls Cheat Sheet -->
      <div class="card">
        <div class="card-title">
          <span>🎮 DASHBOARD BUTTONS CHEAT SHEET</span>
        </div>
        <table class="guide-table">
          <thead>
            <tr>
              <th style="width: 180px;">Button</th>
              <th style="width: 140px;">Type / Style</th>
              <th>What It Does</th>
              <th>When to Click It</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>▶ GRASP QUEUE</strong></td>
              <td><span style="color:#10b981; font-weight:700;">Green (Main)</span></td>
              <td>Freezes target coordinates and executes pick-and-retreat sequence for all queued tubes.</td>
              <td>After selecting tubes in the queue.</td>
            </tr>
            <tr>
              <td><strong>🦾 Ready Pose</strong></td>
              <td><span style="color:#60a5fa; font-weight:700;">Blue</span></td>
              <td>Moves arm to overhead observation height and opens gripper.</td>
              <td>First action when starting the station.</td>
            </tr>
            <tr>
              <td><strong>⚡ Enable Robot</strong></td>
              <td><span style="color:#10b981; font-weight:700;">Green Outline</span></td>
              <td>Sends CAN enable command to arm. The LED indicator turns <strong>glowing green only after the robot driver confirms active torque feedback</strong>.</td>
              <td>If robot was disabled, or after power up.</td>
            </tr>
            <tr>
              <td><strong>🛑 Disable Robot</strong></td>
              <td><span style="color:#ef4444; font-weight:700;">Red Outline</span></td>
              <td>Cuts motor power (limp mode) so arm can be guided by hand. The LED indicator turns <strong>dim gray</strong>.</td>
              <td>When shutting down or hand-positioning.</td>
            </tr>
            <tr>
              <td><strong>🏠 Safe Home</strong></td>
              <td><span style="color:#f59e0b; font-weight:700;">Amber</span></td>
              <td>Releases gripper and smoothly parks arm at zero position via minimum-jerk path.</td>
              <td>When finishing work or aborting safely.</td>
            </tr>
            <tr>
              <td><strong>🎯 Macro Center</strong></td>
              <td><span style="color:#818cf8; font-weight:700;">Purple</span></td>
              <td>Centers camera optical axis directly over the selected tube.</td>
              <td>To inspect a specific tube from above.</td>
            </tr>
            <tr>
              <td><strong>❄️ Reset / Unfreeze</strong></td>
              <td><span style="color:#94a3b8; font-weight:700;">Slate Gray</span></td>
              <td>Unfreezes video feed and returns camera to live scanning mode.</td>
              <td>If you want to cancel a frozen scan.</td>
            </tr>
            <tr>
              <td><strong>⏸ Resume from Pause</strong></td>
              <td><span style="color:#ef4444; font-weight:700;">Red Flashing</span></td>
              <td>Resumes motion if an approach or retreat waypoint has a pause trigger.</td>
              <td>Only visible during programmed pauses.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Navigation Tabs Guide -->
      <div class="card">
        <div class="card-title">
          <span>📑 THE 4 DASHBOARD TABS EXPLAINED</span>
        </div>
        <div class="guide-step-grid" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));">
          <div class="guide-step-card">
            <strong style="color: #60a5fa;">🎮 Visual Grasping</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              Main operational view: live video stream, detected tubes, picking queue, motion controls, and system telemetry log.
            </p>
          </div>
          <div class="guide-step-card">
            <strong style="color: #818cf8;">📍 Waypoints Editor</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              Add, inspect, or delete approach and retreat waypoints (e.g. lift height, tilt angles). Automatically saves to <code>config/waypoints.yaml</code>.
            </p>
          </div>
          <div class="guide-step-card">
            <strong style="color: #34d399;">⚙️ Robot &amp; Camera Config</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              Adjust camera focus, table Z height, pre-grasp clearances, and ready pose coordinates without modifying source code. Saves to <code>config/default.yaml</code>.
            </p>
          </div>
          <div class="guide-step-card">
            <strong style="color: #fbbf24;">📖 User Guide</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              This quick-start reference guide! Always available for operators, lab technicians, and new engineers.
            </p>
          </div>
        </div>
      </div>

      <!-- Frequently Asked Questions -->
      <div class="card" style="margin-bottom: 0;">
        <div class="card-title">
          <span>❓ FREQUENTLY ASKED QUESTIONS &amp; SAFETY</span>
        </div>
        <div class="guide-faq-grid">
          <div class="guide-faq-card">
            <strong style="color: #fff; font-size: 0.9rem;">Q: Why doesn't the robot move when the server starts?</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              For safety, the robot enables holding torque on startup but <strong>holds still</strong> until you explicitly click <strong>"Ready Pose"</strong>.
            </p>
          </div>
          <div class="guide-faq-card">
            <strong style="color: #fff; font-size: 0.9rem;">Q: Can I pick tubes in reverse or custom order?</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              Yes! Tubes are indexed <strong>#1, #2, #3...</strong> from left to right. Click buttons in whatever order you want (e.g. #4, then #2, then #1).
            </p>
          </div>
          <div class="guide-faq-card">
            <strong style="color: #fff; font-size: 0.9rem;">Q: How do I abort motion if something goes wrong?</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              Click <strong>"Safe Home"</strong> to open the gripper and smoothly park the arm at zero position, or click <strong>"Disable Robot"</strong> to immediately cut motor power.
            </p>
          </div>
          <div class="guide-faq-card">
            <strong style="color: #fff; font-size: 0.9rem;">Q: What does the System Telemetry console show?</strong>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">
              Real-time activity logs and error diagnostics. If an error occurs, an active red alert banner appears with clickable stack tracebacks for debugging.
            </p>
          </div>
        </div>
        <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--card-border); font-size: 0.82rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center;">
          <span>📄 Complete documentation file available at: <code>docs/WEB_GUI_USER_GUIDE.md</code></span>
          <button onclick="switchTab('station')" class="btn-main" style="background: transparent; border: 1px solid var(--card-border); color: #cbd5e1; padding: 4px 12px; border-radius: 4px; cursor: pointer;">
            Back to Station ➔
          </button>
        </div>
      </div>
    </div>
  </main>

  <div id="toast" class="toast">Configuration Saved!</div>

  <script>
    let currentState = 'live';
    let currentTubes = [];
    let currentQueue = [];

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (tabId === 'station') {
        document.querySelector('.nav-tabs .tab-btn:nth-child(1)').classList.add('active');
        document.getElementById('tab-station').classList.add('active');
      } else if (tabId === 'waypoints') {
        document.querySelector('.nav-tabs .tab-btn:nth-child(2)').classList.add('active');
        document.getElementById('tab-waypoints').classList.add('active');
        loadWaypoints();
      } else if (tabId === 'config') {
        document.querySelector('.nav-tabs .tab-btn:nth-child(3)').classList.add('active');
        document.getElementById('tab-config').classList.add('active');
        loadConfiguration();
      } else if (tabId === 'guide') {
        document.querySelector('.nav-tabs .tab-btn:nth-child(4)').classList.add('active');
        document.getElementById('tab-guide').classList.add('active');
      }
    }

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
            badgeText.innerText = '⚡ ROBOT ENABLED';
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
            badgeText.innerText = '⚪ ROBOT DISABLED';
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

    function renderTubesBar(tubes, queue) {
      const container = document.getElementById('tubes-bar');
      document.getElementById('tubes-count').innerText = `${tubes.length} tubes detected`;
      if (tubes.length === 0) {
        container.innerHTML = `<span class="empty-queue-msg">Scanning table...</span>`;
        return;
      }

      const queuedIds = queue.map(q => q.tube_id);
      container.innerHTML = tubes.map(t => {
        const isQueued = queuedIds.includes(t.id);
        const cls = isQueued ? 'tube-btn queued' : 'tube-btn';
        return `<button class="${cls}" onclick="enqueueTube(${t.id})">#${t.id}</button>`;
      }).join('');
    }

    function renderQueue(queue) {
      const container = document.getElementById('queue-box');
      if (queue.length === 0) {
        container.innerHTML = `<span class="empty-queue-msg">No tubes queued. Click buttons above to add.</span>`;
        return;
      }
      container.innerHTML = queue.map((item, idx) => {
        return `
          <div class="queue-item">
            <span class="queue-item-info">${idx + 1}. ${item.label}</span>
            <span class="queue-item-coord">Base: (${item.x}, ${item.y}, ${item.z})</span>
            <button class="queue-item-remove" onclick="dequeueItem(${item.queue_uid})">✕</button>
          </div>
        `;
      }).join('');
    }

    async function enqueueTube(tubeId) {
      await fetch('/api/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add', tube_id: tubeId })
      });
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
        badgeText.innerText = '⚡ VERIFYING FEEDBACK...';
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
          ? `<details class="log-trace-details"><summary>🔍 View Stack Trace</summary><pre class="log-trace-pre">${escapeHtml(trace)}</pre></details>`
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
        badge.innerText = `📍 ${total} WPs (${appCount || 0} App / ${retCount || 0} Ret)`;
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
        const isAbs = ('x' in wp || 'y' in wp || 'z' in wp);
        const modeBadge = isAbs
          ? `<span class="wp-badge-mode wp-badge-abs">ABS</span>`
          : `<span class="wp-badge-mode wp-badge-rel">REL</span>`;

        const posText = isAbs
          ? `X: ${wp.x ?? 0}m, Y: ${wp.y ?? 0}m, Z: ${wp.z ?? 0}m`
          : `dx: ${wp.dx ?? 0}m, dy: ${wp.dy ?? 0}m, dz: ${wp.dz ?? 0}m`;

        const rpyText = `RPY: [${wp.roll ?? 0}, ${wp.pitch ?? 0}, ${wp.yaw ?? 0}] rad`;
        const durText = `${wp.duration ?? 2.0}s`;
        const gripText = wp.gripper ? ` | Grip: ${wp.gripper}` : '';
        const pauseText = wp.pause ? ` | ⏸ Pause` : '';

        return `
          <div class="wp-item-row">
            <div class="wp-item-desc">
              <div><strong>#${idx + 1}</strong> ${modeBadge} <span style="font-family:monospace; color:#38bdf8;">${posText}</span></div>
              <div style="font-size:0.75rem; color:var(--text-muted);">${rpyText} | ${durText}${gripText}${pauseText}</div>
            </div>
            <button class="queue-item-remove" onclick="removeWaypoint('${type}', ${idx})" title="Remove waypoint">✕</button>
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

      } catch (err) {
        console.error("Failed to load config", err);
      }
    }

    async function saveConfiguration() {
      const payload = {
        camera: {
          id: document.getElementById('cfg-cam-id').value,
          focus: parseInt(document.getElementById('cfg-cam-focus').value) || 0,
          autofocus: document.getElementById('cfg-cam-autofocus').value === 'true',
          intrinsics_file: document.getElementById('cfg-cam-intrinsics').value,
        },
        grasp_pipeline: {
          grasp: {
            table_z_m: parseFloat(document.getElementById('cfg-table-z').value) || 0.17,
            pregrasp_offset_m: parseFloat(document.getElementById('cfg-pregrasp-offset').value) || 0.15,
            insertion_depth_m: parseFloat(document.getElementById('cfg-insertion-depth').value) || 0.0,
          }
        },
        calibration: {
          hand_eye_file: document.getElementById('cfg-he-file').value,
          hand_eye_compensation_m: {
            x: parseFloat(document.getElementById('cfg-he-comp-x').value) || 0.0,
            y: parseFloat(document.getElementById('cfg-he-comp-y').value) || 0.0,
            z: parseFloat(document.getElementById('cfg-he-comp-z').value) || 0.0,
          }
        },
        robot: {
          ready_pose: {
            x: parseFloat(document.getElementById('cfg-rp-x').value) || 0.3,
            y: parseFloat(document.getElementById('cfg-rp-y').value) || 0.0,
            z: parseFloat(document.getElementById('cfg-rp-z').value) || 0.3,
            pitch: parseFloat(document.getElementById('cfg-rp-pitch').value) || 0.7,
          },
          gripper: {
            ready_open_m: parseFloat(document.getElementById('cfg-gripper-ready-open').value) || 0.055,
            fixed_close_m: document.getElementById('cfg-gripper-fixed-close').value !== "" ? parseFloat(document.getElementById('cfg-gripper-fixed-close').value) : null,
            close_offset_m: parseFloat(document.getElementById('cfg-gripper-close-offset').value) || 0.010,
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
          showToast("Configuration Saved to default.yaml!");
        } else {
          alert("Error saving: " + data.error);
        }
      } catch (err) {
        alert("Failed to save config: " + err);
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

    // Auto-poll state every 200ms
    setInterval(updateState, 200);
    updateState();
    loadWaypoints();
  </script>
</body>
</html>
"""


class MainIndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
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

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        vision_worker.running = False
        robot_worker.running = False
        vision_worker.join(timeout=1.0)
        robot_worker.join(timeout=1.0)
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
