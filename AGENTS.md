# reBot Arm B601-DM — System Architecture, APIs & Developer Guidelines (AGENTS.md)

> **Audience**: AI Agents, Robotics Engineers, Control & Perception Developers.  
> **Repository**: `Pick-Place-B601DM`  
> **Hardware**: reBot Arm B601-DM (6-DOF Manipulator + 1-DOF Parallel Gripper, Damiao Smart Actuators, USB-to-CAN Serial Bridge at 921,600 baud, Fixed-Focus 2D Monocular USB Webcam).  
> **Key Frameworks**: Python 3.10, Pinocchio (Spatial Kinematics & Rigid Body Dynamics), Ultralytics YOLO, OpenCV, Tornado (Async Web GUI).

---

## Table of Contents
1. [Cross-Folder Architecture & System Data Flow](#1-cross-folder-architecture--system-data-flow)
2. [End-to-End Perception & Motion Execution Pipeline](#2-end-to-end-perception--motion-execution-pipeline)
3. [URDF, Kinematics & Coordinate Frame Mathematics](#3-urdf-kinematics--coordinate-frame-mathematics)
4. [Shared Wrappers & Utility Inventory](#4-shared-wrappers--utility-inventory)
   - [4.1 `drivers/` (Hardware Abstraction Layer)](#41-drivers-hardware-abstraction-layer)
   - [4.2 `calibration/` (Optical & Hand-Eye Calibration)](#42-calibration-optical--hand-eye-calibration)
   - [4.3 `utils/` (Core Transformation & Perception Libraries)](#43-utils-core-transformation--perception-libraries)
5. [Configuration & Waypoints YAML Schemas](#5-configuration--waypoints-yaml-schemas)
   - [5.1 `config/default.yaml` Master Schema](#51-configdefaultyaml-master-schema)
   - [5.2 `config/waypoints.yaml` Schema](#52-configwaypointsyaml-schema)
6. [Web Grasp GUI Architecture & Interactive Tour](#6-web-grasp-gui-architecture--interactive-tour)
7. [Hardware Safety Protocols & Operational Rules](#7-hardware-safety-protocols--operational-rules)
8. [Testing, Verification & CLI Reference](#8-testing-verification--cli-reference)

---

## 1. Cross-Folder Architecture & System Data Flow

The codebase is organized into modular layers that separate low-level motor communication from high-level perception, motion planning, and asynchronous user interfaces.

```
                              +---------------------------------------+
                              |         Operator Web GUI              |
                              |      (scripts/web_grasp_gui.py)       |
                              |  Tornado REST API + WebSocket/Polling |
                              +-------------------+-------------------+
                                                  |
                                                  v
                              +---------------------------------------+
                              |         Motion Planning & Execution   |
                              |       (scripts/fixed_path_grasp.py)   |
                              |       (scripts/move_arm_ik.py)        |
                              |     Waypoint & Queue State Machine    |
                              +---------+-------------------+---------+
                                        |                   |
                     Ray-Plane Table    |                   | Trajectory
                     Intersection Pose  |                   | Generation
                                        v                   v
+-------------------------------+-------------------+   +------------------------------------+
|       Perception Layer        |  Kinematics Core  |   |        Robot & Gripper HAL         |
|  drivers/camera/webcam.py     |  Pinocchio IK/FK  |   |  drivers/robot/grasp_driver.py     |
|  utils/ordinary_grasp.py      |  sdk/kinematics/  |   |  - Register Probing without torque |
|  utils/yolo_utils.py          |  - Forward Kin.   |   |  - Dual Mode (Arm PosVel, Grip MIT)|
|  calibration/hand_eye.py      |  - CLIK Solver    |   |  - State feedback streaming        |
+---------------+---------------+---------+---------+   +-----------------+------------------+
                ^                         |                               |
                | T_tcp2base (FK)         | Joint Angles q                | Motor Setpoints
                +-------------------------+                               v
                                                        +------------------------------------+
                                                        |      reBotArm_control_py SDK       |
                                                        |  sdk/reBotArm_control_py/actuator  |
                                                        |  Controller.from_dm_serial         |
                                                        +-----------------+------------------+
                                                                          |
                                                                          v 921,600 baud
                                                        +------------------------------------+
                                                        |     Physical Robot Hardware        |
                                                        |     /dev/ttyACM* (Damiao CAN)      |
                                                        |  Joints 1-6 (4340P/4310) + Gripper |
                                                        +------------------------------------+
```

### Folder Responsibilities & Cross-Dependencies

| Directory | Responsibilities & Boundaries | Key Consumers / Callers |
| :--- | :--- | :--- |
| `drivers/` | Device drivers for camera (`webcam.py`, `realsense.py`, `orbbec_gemini2.py`) and robot gripper abstraction (`grasp_driver.py`). Provides uniform interfaces without leaking vendor quirks. | `scripts/`, `utils/camera_utils.py` |
| `sdk/` | Low-level C++/Python SDK (`reBotArm_control_py`) for Damiao/RS actuators, Pinocchio kinematics models, gravity compensation controllers, and Cartesian path planners. **Do not modify without explicit approval.** | `drivers/robot/`, `scripts/` |
| `calibration/` | Eye-in-hand calibration solver (`hand_eye.py`) and ArUco pose estimation (`aruco_pose.py`). Computes $T_{\text{cam}\to\text{gripper}}$ and camera intrinsics. | `scripts/collect_handeye_eih.py`, `scripts/calibrate_camera.py` |
| `models/` | Neural network checkpoints (`best_with_green_rack.pt`, `best.pt`, `yolov8s-world.pt`). | `utils/yolo_utils.py`, `scripts/web_grasp_gui.py` |
| `utils/` | Mathematics, coordinate transformations, 2D ray-plane intersection, Jetson CUDA memory bounding, and vibration analysis. Pure shared algorithms. | `scripts/`, `tests/` |
| `config/` | System configuration (`default.yaml`), waypoints file (`waypoints.yaml`), and camera/hand-eye calibration `.npz` storage. | Loaded globally by all scripts. |
| `scripts/` | Executable user workflows: Web GUI (`web_grasp_gui.py`), CLI grasp automation (`fixed_path_grasp.py`), Cartesian mover (`move_arm_ik.py`), verification tools. | End-user / System services. |
| `tests/` | Unittest suite covering hardware probing, ACK recovery, 2D ray-plane intersection, waypoint schemas, and Web GUI APIs. | Automated CI / Pre-commit checks. |
| `docs/` | User documentation, hardware accuracy audits, camera focus guides, and Web GUI operating manuals. | Reference documentation. |

---

## 2. End-to-End Perception & Motion Execution Pipeline

When an object (e.g. Falcon tube) is selected for grasping, data traverses the pipeline through 6 mathematical stages:

```
[Webcam Frame 1280x720]
         |
         v
1. YOLO Detection (utils/yolo_utils.py)
   --> Bounding box (u1, v1, u2, v2), Class Name, Confidence
   --> Oriented Bounding Box (rect_points), Principal Axis (short_edge_points)
         |
         v
2. 2D Planar Ray-Plane Intersection (utils/ordinary_grasp.py)
   --> Camera ray: d_c = [(u - c_x)/f_x, (v - c_y)/f_y, 1.0]^T
   --> Base ray:   d_b = T_cam2base * d_c,  o_b = T_cam2base * [0, 0, 0, 1]^T
   --> Intersection at table plane Z = table_z_m:
         t = (table_z_m - o_b[z]) / d_b[z]
         P_base = o_b + t * d_b
         z_cam = (inv(T_cam2base) * P_base)[z]
         |
         v
3. Eye-in-Hand Coordinate Composition (utils/camera_utils.py)
   --> T_cam2base = T_compensation @ T_tcp2base @ T_hand_eye
   --> T_tcp2base is queried live via Forward Kinematics (FK)
         |
         v
4. Gripper Tool-Frame Alignment (utils/transforms.py)
   --> Approach Axis (tool-forward X): directed downward toward table
   --> Opening Axis (tool Y): aligned with gripper jaw opening
   --> Z Axis: right-handed completion
   --> Canonicalization: chooses branch minimizing absolute roll
         |
         v
5. Waypoint & Trajectory Resolution (scripts/web_grasp_gui.py / fixed_path_grasp.py)
   --> Pre-grasp (P_target + dz_pregrasp) -> Final Grasp (P_target + z_offset)
   --> Gripper Closing: commanded to fixed_close_m (or distance - offset)
   --> Sequential Retreat: relative offsets (dx, dy, dz) or absolute poses (x, y, z, roll, pitch, yaw)
   --> Safe Return: Ready Pose
         |
         v
6. Inverse Kinematics & Motor Execution (sdk/kinematics/inverse_kinematics.py)
   --> Closed-Loop Inverse Kinematics (CLIK) with adaptive damping:
         delta_q = J^T (J J^T + lambda^2 I)^(-1) * e
   --> Setpoint interpolation (500 Hz) via Damiao POS_VEL / MIT packets
```

---

## 3. URDF, Kinematics & Coordinate Frame Mathematics

### 3.1 URDF Kinematic Structure (`ReBot_Arm_DM.urdf`)
- **Location**: `sdk/reBotArm_control_py/urdf/DM/urdf/ReBot_Arm_DM.urdf`
- **Total Generalized Coordinates (`nq`)**: `8`
- **Degrees of Freedom (`nv`)**: `8`
- **Controlled Arm Joints (`arm_joints`)**: `6` (prefix configuration slice `q[:6]`)
- **End-Effector Frame**: `end_link` (Frame ID queried via `model.getFrameId("end_link")`)
- **TCP Structural Offset (`end_joint`)**:
  Fixed joint linking `link6` to `end_link`:
  $$\text{origin xyz} = [0,\ 0,\ 0.15539]\,\text{m},\quad \text{rpy} = [0,\ -1.5708,\ 3.1415]\,\text{rad}$$
- **Gripper Finger Decoupling**:
  Downstream prismatic joints (`finger_left` and `finger_right`, range `0` to `0.0285` m). Downstream motion of fingers alters **zero** coordinates of the `end_link` kinematic frame. Verified in unit tests to $< 10^{-12}$ m position error.

### 3.2 Joint Limits & Actuator Specs

| Joint | Motor ID | Feedback ID | Model | Joint Range (rad) | Range (deg) | Max Velocity | Default Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `joint1` | `0x01` | `0x11` | Damiao 4340P | $[-2.618, +2.618]$ | $[-150^\circ, +150^\circ]$ | $200\,\text{rad/s}$ | `posvel` |
| `joint2` | `0x02` | `0x12` | Damiao 4340P | $[-1.832, +1.832]$ | $[-105^\circ, +105^\circ]$ | $200\,\text{rad/s}$ | `posvel` |
| `joint3` | `0x03` | `0x13` | Damiao 4340P | $[-2.618, +2.618]$ | $[-150^\circ, +150^\circ]$ | $200\,\text{rad/s}$ | `posvel` |
| `joint4` | `0x04` | `0x14` | Damiao 4310 | $[-2.618, +2.618]$ | $[-150^\circ, +150^\circ]$ | $200\,\text{rad/s}$ | `posvel` |
| `joint5` | `0x05` | `0x15` | Damiao 4310 | $[-2.618, +2.618]$ | $[-150^\circ, +150^\circ]$ | $200\,\text{rad/s}$ | `posvel` |
| `joint6` | `0x06` | `0x16` | Damiao 4310 | $[-3.140, +3.140]$ | $[-180^\circ, +180^\circ]$ | $200\,\text{rad/s}$ | `posvel` |
| `gripper` | `0x07` | `0x17` | Damiao 4310 | $[0.0, 5.0]\,\text{rad}$ | $[0^\circ, 286^\circ]$ | Stroke: 0–9 cm | `mit` |

### 3.3 Rotation Mathematics & Conventions
- **Euler Convention**: Intrinsic ZYX rotation order (equivalent to extrinsic XYZ rotations):
  $$R(\phi, \theta, \psi) = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)$$
  where $\phi = \text{roll}$ (X), $\theta = \text{pitch}$ (Y), $\psi = \text{yaw}$ (Z).
- **Pinocchio Compatibility**: `pin.rpy.rpyToMatrix(roll, pitch, yaw)` and `pin.rpy.matrixToRpy(rot)` map directly to this convention.
- **Symmetric Parallel Gripper Invariance**: Rotating $180^\circ$ around tool X ($R_{\text{alt}} = R \cdot R_x(\pi)$) results in an identical physical grasp. `canonicalize_parallel_gripper_tcp_rotation()` automatically chooses the branch with smaller absolute roll to prevent unnecessary wrist wrap-around.

---

## 4. Shared Wrappers & Utility Inventory

Developers must **reuse** these existing helpers rather than re-implementing math or serial routines.

### 4.1 `drivers/` (Hardware Abstraction Layer)

#### `drivers/robot/grasp_driver.py`
- `resolve_dm_serial_channel(arm) -> str`:  
  *Behavior*: Probes candidate `/dev/ttyACM*` and `/dev/ttyUSB*` ports using **read-only** register reads (Register 10: Control Mode). Verifies all 7 motors reply before selecting an adapter. **Never enables motors during probing.** Releases all file descriptors cleanly.
- `selected_arm_config(repo_root=None) -> SelectedArmConfig`:  
  *Returns*: `SelectedArmConfig(arm_type="dm", controller_mode="posvel")`.
- `class GraspDriver(arm, controller, gripper_config=None, repo_root=None)`:
  - `start() -> None`: Initializes serial communication, retries mode ACKs up to 3 times, validates finite non-zero feedback, primes Cartesian setpoints, and attaches 500 Hz control loop.
  - `open_gripper(distance_m=0.09, timeout=3.0) -> None`: Drives gripper to linear opening (0.0 to 0.09 m). Automatically transitions to `_STATE_IDLE` upon arrival to eliminate motor holding dither and wrist vibration.
  - `grasp(force=None, close_torque=None, kd_close=None, timeout=5.0) -> bool`: Closes jaws with compliant velocity/torque control. Detects stall ($|v| < 0.05$ rad/s) as successful object contact (`True`), or hard stop ($v \to 0$ at mechanical limit) as empty grasp (`False`).
  - `release_gripper(timeout=4.0) -> None`: Opens jaws and relaxes motor.
  - `get_gripper_state() -> tuple[float, float, float]`: Returns `(pos_rad, vel_rad_s, torq_nm)`.
  - `get_tcp_pose() -> np.ndarray`: Computes live $4 \times 4$ SE(3) matrix of the TCP in base frame via Pinocchio FK.

#### `drivers/camera/base.py` & `webcam.py`
- `class WebcamDriver(CameraDriver)`:
  - `open() -> None`: Opens `/dev/video*` via `v4l2` backend, negotiates 1280x720 MJPG at 30 FPS, applies absolute lens focus (e.g. `focus=0`), and verifies readback.
  - `get_frame() -> tuple[Optional[np.ndarray], None]`: Returns `(bgr_image, None)` (depth is explicitly `None`).
  - `K -> np.ndarray`: Returns $3 \times 3$ intrinsic matrix (loaded from calibrated `.npz` or generated).
  - `D -> np.ndarray`: Returns $1 \times 5$ distortion coefficients.

### 4.2 `calibration/` (Optical & Hand-Eye Calibration)

#### `calibration/hand_eye.py`
- `class HandEyeCalibrator(mode=CalibMode.EYE_IN_HAND, method="TSAI")`:
  - `add_sample(T_gripper2base: np.ndarray, T_marker2cam: np.ndarray) -> None`: Stashes FK pose and detected ArUco pose.
  - `calibrate(min_samples=5) -> CalibResult`: Solves $T_{\text{cam}\to\text{gripper}}$ using `cv2.calibrateHandEye` such that $T_{\text{marker}\to\text{base}} = T_{\text{gripper}\to\text{base}} \cdot T_{\text{cam}\to\text{gripper}} \cdot T_{\text{marker}\to\text{cam}}$.
  - `save(result, path) / load(path)`: Serializes/deserializes `.npz` calibration artifacts.

#### `calibration/aruco_pose.py`
- `class ArUcoDetector(marker_length_m, dict_id=0, target_marker_id=None)`:
  - `detect(bgr, K, D) -> Optional[np.ndarray]`: Detects AprilTag / ArUco markers and returns $4 \times 4$ homogeneous matrix $T_{\text{marker}\to\text{cam}}$.

### 4.3 `utils/` (Core Transformation & Perception Libraries)

#### `utils/transforms.py`
- `pose6d_to_mat4(x, y, z, rx, ry, rz, degrees=False) -> np.ndarray`:  
  Converts Cartesian 6D coordinates into a $4 \times 4$ matrix using intrinsic ZYX Euler rotations.
- `mat4_to_pose6d(T: np.ndarray) -> tuple[float, float, float, float, float, float]`:  
  Decomposes $4 \times 4$ SE(3) matrix into $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$ in radians.
- `rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray`:  
  Orthogonalizes $R$ via SVD and extracts intrinsic ZYX Euler angles with gimbal lock singularity checks.
- `canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray`:  
  Returns the equivalent orientation branch ($R$ vs $R \cdot R_x(\pi)$) minimizing absolute roll.
- `grasp_axes_to_rebot_tcp_rotation(grip_axis, open_axis, approach_axis) -> np.ndarray`:  
  Maps vision grasp frame (approach pointing toward object) into reBotArm TCP coordinates (tool-forward X, opening Y, normal Z).
- `transform_grasp_pose_to_base(position_cam, tcp_rotation_cam, T_cam2base, pregrasp_offset_m, insertion_depth_m=0.0)`:  
  Converts camera-frame grasp pose into base-frame $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$ tuples for both grasp and pregrasp.

#### `utils/ordinary_grasp.py`
- `estimate_grasps(results, depth_mm, K, depth_quantile=0.75, T_cam2base=None, table_z_m=0.0) -> list[GraspPose]`:  
  Performs 2D Ray-Plane Table Intersection when `depth_mm is None`. Computes spatial coordinates, oriented bounding boxes, and approach vectors.
- `draw_grasp(image: np.ndarray, grasp: GraspPose) -> None`:  
  Draws 2D bounding boxes, rotated oriented bounding boxes, center dots, and coordinate text overlays.

#### `utils/camera_utils.py`
- `load_config(path: str | Path) -> dict[str, Any]`: Loads and parses YAML files safely.
- `load_hand_eye(project_root, cam_type, filename="hand_eye.npz") -> tuple[Optional[np.ndarray], Optional[str]]`: Loads hand-eye calibration matrix from disk.
- `hand_eye_compensation_matrix(cfg: dict[str, Any]) -> np.ndarray`: Returns $4 \times 4$ translation matrix from `calibration.hand_eye_compensation_m`.
- `compose_cam_to_base_transform(T_tcp2base, T_hand_eye, cfg) -> np.ndarray`:  
  Computes $T_{\text{cam}\to\text{base}} = T_{\text{compensation}} \cdot T_{\text{tcp}\to\text{base}} \cdot T_{\text{hand\_eye}}$.

#### `utils/yolo_utils.py`
- `configure_jetson_inference() -> None`:  
  Sets `CUDNN_CONV_WSCAP_DBG=128` and `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64` when `/etc/nv_tegra_release` is detected to prevent Jetson NvMap OOM assertions.
- `warmup_yolo(model, frame, yolo_opts) -> None`:  
  Exercises GPU memory before robot initialization. Raises `DetectorMemoryError` on failure so motors are never enabled with an unhealthy vision model.
- `load_yolo(cfg, project_root, ...) -> tuple[Optional[Any], dict[str, Any]]`:  
  Standardized YOLO model loader with fallback and parameter binding.

#### `utils/vibration_utils.py`
- `class VibrationDataset`: Container for synchronized time, joint encoder setpoints, actual positions, velocities, motor torques, and Cartesian FK error metrics.
- `compute_cartesian_trajectories(q_cmd, q_act, repo_root=None)`: Computes commanded and actual Cartesian paths using Pinocchio.
- `run_fft_analysis(...)`: Computes Power Spectral Density and FFT frequency peaks to identify mechanical resonances.

---

## 5. Configuration & Waypoints YAML Schemas

### 5.1 `config/default.yaml` Master Schema

```yaml
camera:
  type: webcam                 # Camera driver: 'webcam' | 'realsense' | 'orbbec_gemini2'
  id: 4                        # V4L2 device index (e.g. /dev/video4)
  serial: null                 # Serial number string (for depth cameras)
  color_width: 1280            # Horizontal image resolution
  color_height: 720            # Vertical image resolution
  fps: 30                      # Camera capture framerate
  fourcc: MJPG                 # Pixel format: 'MJPG' or 'YUYV'
  buffer_size: 2               # V4L2 buffer queue depth
  autofocus: false             # Must be false when using calibrated fixed focus
  focus: 0                     # Absolute lens focus index (0 = infinity/fixed macro)
  intrinsics_file: config/calibration/webcam/intrinsics_Nov08_Focus_0.npz

calibration:
  hand_eye_file: hand_eye_test_manually_focus0_nov08.npz  # Active eye-in-hand matrix
  aruco:
    marker_length_m: 0.1       # Physical black-square side length (meters)
    dict_id: 20                # cv2.aruco dictionary: 20 = DICT_APRILTAG_36h11
    target_marker_id: 0        # Expected marker ID
  hand_eye_method: TSAI        # Solver algorithm: 'TSAI' | 'PARK' | 'HORAUD' | 'ANDREFF'
  hand_eye_compensation_m:     # Empirical translation bias offsets (meters)
    x: 0.02
    y: 0.01
    z: 0.00

detection:
  conf_threshold: 0.50         # Minimum YOLO confidence score
  iou_threshold: 0.45          # NMS IoU threshold

robot:
  gripper:
    ready_open_m: 0.055        # Jaw opening distance at Ready Pose (meters)
    close_offset_m: 0.010      # Distance subtracted from detected object width (meters)
    fixed_close_m: 0.030       # Override: close to fixed stroke (0.03 = 3.0cm)
    dm:
      angle_open: 4.0          # Damiao gripper motor target angle (rad)
      counterclockwise: true   # Motor rotation direction
      tau_max: 1.5             # Max allowed motor torque (Nm)
      close_torque: 1.0        # Closing torque limit (Nm)
      default_force: 0.30      # Continuous holding torque (Nm)
  ready_pose:                  # Primary observation / scanning pose (base frame)
    x: 0.30                    # X coordinate (meters)
    y: 0.00                    # Y coordinate (meters)
    z: 0.30                    # Z coordinate (meters)
    roll: 0.00                 # Extrinsic roll (rad)
    pitch: 0.70                # Extrinsic pitch (rad)
    duration: 3.0              # Transition duration (seconds)

yolo:
  model_name: best_with_green_rack.pt # File inside models/
  device: '0'                  # PyTorch device: '0' for CUDA GPU, 'cpu' for CPU
  use_world: true              # Open-vocabulary YOLO-World flag
  custom_classes:
    - yellow screwdriver
    - blue cap with transparent centrifuge tube inside a container
    - blue cube

grasp_pipeline:
  infer_every_live: 1          # Inference decimation rate
  grasp:
    table_z_m: 0.17            # Physical height of table plane in base coordinates (meters)
    depth_quantile: 0.50       # Quantile used if depth camera is present
    pregrasp_offset_m: 0.15    # Stand-off distance along tool-approach axis (meters)
    insertion_depth_m: 0.00    # Extra plunge depth beyond detected surface (meters)
    grasp_z_offset_m: -0.02    # Global Z offset applied to final grasp point (meters)
    min_base_z_m: 0.00         # Hard safety floor: manipulator never drives below this Z
```

### 5.2 `config/waypoints.yaml` Schema

The waypoints file supports **Relative Cartesian**, **Absolute Cartesian**, and **6-Axis Joint Space** formats:

```yaml
waypoints:
  approach:
    # 1. Absolute Cartesian Pose
    - x: 0.35
      y: 0.00
      z: 0.30
      roll: 0.0
      pitch: 0.7
      yaw: 0.0
      duration: 2.0
      gripper: open            # 'open', 'close', or stroke in meters (e.g. 0.04)

  retreat:
    # 2. Relative Cartesian Waypoint (offsets from grasp point)
    - dx: 0.00
      dy: 0.00
      dz: 0.10                 # Lift vertically +10 cm
      duration: 2.0

    # 3. Absolute Cartesian Waypoint with Orientation
    - x: 0.35
      y: 0.30
      z: 0.18
      roll: -0.8
      pitch: 0.0
      yaw: 1.2
      duration: 3.0
      pause: false             # Set true to hold pose until operator presses SPACE

    # 4. Joint Space Waypoint (6-Axis Radian Setpoints)
    - joints: [0.125, -0.650, -0.850, 0.020, 0.450, -0.100]
      duration: 2.5

motion_parameters:
  fixed_grasp_rpy:
    roll: 0.0
    pitch: 0.0
    yaw: 0.0
  pregrasp:
    enabled: true
    dx: 0.0
    dy: 0.0
    dz: 0.10
    duration: 5.0
  grasp:
    move_duration: 2.0
    ready_open_m: 0.055
    close_offset_m: 0.010
    fixed_close_m: 0.030
```

---

## 6. Web Grasp GUI Architecture & Interactive Tour

`scripts/web_grasp_gui.py` is an asynchronous Tornado application providing a live visual control station.

### 6.1 State Machine & Threading Model
- **Main Thread**: Runs Tornado `IOLoop` serving REST API endpoints (`/api/status`, `/api/action`, `/api/queue`, `/api/config`, `/api/teach`), live MJPEG camera streams (`/video_feed`), and Web GUI assets.
- **Worker Thread (`RobotWorker`)**: Dedicated daemon thread with a task queue (`queue.Queue`). Consumes commands sequentially:
  - `grasp`: Executes sequential Falcon tube grasping pipeline.
  - `macro_center`: Centers camera over an object.
  - `ready_pose`: Moves arm smoothly to Ready Pose.
  - `safe_home`: Drives arm through a safe intermediate waypoint to resting home pose.
  - `emergency_stop`: Immediate motion abort sequence.
  - `test_gripper`: Tests jaw clamping width.
- **Abort Event (`threading.Event`)**: Set immediately by `/api/action` when `action=emergency_stop` is requested. The `RobotWorker` checks `abort_event.is_set()` before and between every motion trajectory segment.

### 6.2 Frontend Tabs & Design System
The Web GUI adheres strictly to the **Apple macOS Human Interface Guidelines (HIG)** and **Linear** dark-mode visual design system:

- **Theme Palette**:
  - Background: `#0B0F19` (Deep slate)
  - Card Surfaces: `#131A29` (Subtle 1px border: `rgba(255, 255, 255, 0.08)`)
  - Primary Action / Selection: `#0A84FF` (macOS Accent Blue)
  - Success / Active State: `#30D158` (iOS / HIG Emerald)
  - Emergency Stop / Danger: `#FF453A` (Vibrant HIG Crimson)
  - Typography: Monospace tabular numerals (`font-variant-numeric: tabular-nums;`) for all live coordinate telemetry.
  - Glyph Constraint: **Zero non-ASCII emoji characters**; clean typographic labels only.

- **The 4 Operational Tabs**:
  1. `Visual Grasping` (`#tab-station`): Live camera feed with numbered Falcon tube badges (sorted strictly left-to-right), custom sequential pick queue, hardware LED torque indicator, and motion control triggers.
  2. `Waypoints Editor` (`#tab-waypoints`): Live interactive list of approach and retreat waypoints from `waypoints.yaml`, with forms to append Cartesian or Joint poses, reorder, or delete entries.
  3. `Robot & Camera Config` (`#tab-config`): Live editor for camera parameters, optical intrinsics, grasp plane offsets, ready pose, and gripper stroke. Automatically synchronizes changes to `default.yaml` on disk.
  4. `Teach & Record` (`#tab-teach`): Compliant zero-torque gravity compensation mode with live 6-axis FK HUD, slide mode selector (`XYZ RPY` vs `Joint Angles`), snapshot staging board, and 1-click export to approach, retreat, or ready pose.

### 6.3 Interactive Spotlight Product Tour
Triggered by the `[ ? Guide ]` button in the header bar.
- **Transparent Cutout Technique**:
  - `#tour-overlay`: Container with `background: transparent; pointer-events: auto; overflow: hidden;` and **zero** `backdrop-filter: blur()`.
  - `#tour-spotlight-box`: Positioned directly over target element via `getBoundingClientRect()`. Uses `background: transparent; pointer-events: none;` with an enormous outer shadow:
    ```css
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.78), 0 0 20px rgba(10, 132, 255, 0.45);
    border: 2px solid var(--primary);
    ```
  - This guarantees highlighted buttons and text inside the cutout remain **100% sharp, bright, and completely unblurred**, while all other UI elements outside the box are dimmed.

---

## 7. Hardware Safety Protocols & Operational Rules

> [!CAUTION]
> **MANDATORY MOTOR SAFETY PROTOCOL**
> Physical robot actuators are capable of rapid accelerations and high torque. The following rules are hard invariants across all code paths.

1. **Homing Before Disable (`safe_home` Invariant)**:
   - Motors must **NEVER** be abruptly disabled while the arm is extended in the air. Abruptly releasing torque causes the arm to collapse under gravity, potentially damaging joint gearboxes, the camera, or the table.
   - Any routine disabling motors (e.g. GUI Disable toggle, script exit, Teach Mode exit, or Emergency Stop) **MUST FIRST** execute `safe_home()`:
     1. Move to Intermediate Clearing Pose: $(x=0.30, y=0.00, z=0.30, \text{pitch}=0.70)$.
     2. Move to Rest Home Pose: $(x=0.25, y=0.00, z=0.15, \text{pitch}=0.00)$.
     3. Open Gripper to Ready Width: `open_gripper(ready_open_m)`.
     4. Release Motor Torque: Disable motors only after arm is fully seated.
2. **Emergency Stop (E-Stop) Execution Order**:
   The E-Stop handler must execute strictly in this order:
   ```
   1. Set abort_event to interrupt active trajectories.
   2. Clear picking queue and pending motion queue.
   3. Send emergency stop signal to active controller.
   4. Execute safe_home() trajectory.
   5. Open gripper to ready width (release grasped object safely).
   6. Command motor hardware disable (torque released).
   ```
3. **Read-Only Adapter Probing**:
   Never send motor enable commands (`enable()`) during serial port discovery. Always use `resolve_dm_serial_channel()`, which reads Register 10 via `Controller.from_dm_serial` without moving or powering actuators.
4. **Never Modify `sdk/` or `drivers/` Without User Approval**:
   `sdk/reBotArm_control_py` and `drivers/robot/grasp_driver.py` contain hardware-verified motor communication timing, CAN baud rate negotiation, and PID loops. Do not alter them without explicit authorization from the user.
5. **Always Verify in `--dry-run` First**:
   Always run new features or scripts with `--dry-run` to verify kinematics, state transitions, and API responses before physical motor deployment.

---

## 8. Testing, Verification & CLI Reference

### 8.1 Automated Test Suite Execution
The repository includes a complete unittest suite (`64 tests`) covering camera focus, serial connection probing, ACK retries, 2D ray-plane geometry, and Web GUI APIs.

To execute tests cleanly, always set `PYTHONPATH` to prioritize `./scripts` and the internal SDK:

```bash
# Run the entire test suite (all 64 tests)
PYTHONPATH="sdk/reBotArm_control_py:." python3 -m unittest discover -s tests -v

# Run only Web Grasp GUI unit tests (30 tests)
PYTHONPATH="sdk/reBotArm_control_py:." python3 -m unittest tests/test_web_grasp_gui.py -v

# Run robot connection and serial probe tests
PYTHONPATH="sdk/reBotArm_control_py:." python3 -m unittest tests/test_robot_connection.py -v

# Run camera focus and lens verification tests
PYTHONPATH="sdk/reBotArm_control_py:." python3 -m unittest tests/test_camera_focus.py -v
```

### 8.2 Kinematics & URDF Verification Tool
To run numeric forward kinematics, inverse kinematics, and round-trip consistency checks:

```bash
python3 scripts/verify_kinematics_urdf.py
```
*Expected Output*: Position error $< 0.05$ mm, rotation error $< 0.001$ rad across all poses.

### 8.3 Launching the Web Grasp Station

```bash
# Dry-run mode (Safe simulation: full UI, synthetic tubes, simulated motions)
python3 scripts/web_grasp_gui.py --dry-run --port 8000

# Live physical robot mode (Connects to /dev/ttyACM* and physical camera)
python3 scripts/web_grasp_gui.py --port 8000
```

### 8.4 Command-Line Scripts Reference

```bash
# 1. Quick read-only motor connection verification (no motion)
python3 scripts/check_robot_connection.py

# 2. Interactive Cartesian arm mover
python3 scripts/move_arm_ik.py 0.35 0.00 0.28 0.0 0.6 0.0 --dry-run

# 3. CLI Visual Grasping with OpenCV preview
python3 scripts/fixed_path_grasp.py --dry-run

# 4. Camera Intrinsics Calibration (Checkerboard)
python3 scripts/calibrate_camera.py

# 5. Eye-in-Hand Calibration Collection (Manual hand-guiding)
python3 scripts/collect_handeye_eih.py --manual
```
