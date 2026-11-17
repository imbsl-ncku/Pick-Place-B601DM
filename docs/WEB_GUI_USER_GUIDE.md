# reBot Arm B601 — Web Picking Station User Guide
*A quick and simple guide for operators, lab technicians, and new engineers.*

---

## 1. Quick 30-Second Overview

Welcome to the **reBot Arm B601 Picking Station**!

This automated workstation uses an overhead camera and AI vision to detect Falcon centrifuge test tubes sitting in a rack and pick them up one by one using a 6-axis robotic arm with an electric parallel gripper.

```
       [ Overhead Camera ]  -->  Finds tubes in rack & numbers them (#1, #2, #3...)
              |
              v
       [ Web Dashboard ]    -->  You select which tubes to pick & click GRASP
              |
              v
       [ reBot Arm B601 ]   -->  Moves to tube, picks it up, lifts, and tilts safely!
```

---

## 2. 🚀 Quick Start: How to Pick Tubes in 3 Steps

You only need 3 simple steps to pick tubes:

### Step 1: Move Arm to Ready Position 🦾
- When you first open the webpage, the robot is enabled but **holds still for safety**.
- Click the blue **"Ready Pose"** button.
- The robotic arm will smoothly move above the rack and open its gripper, giving the camera an unobstructed view of the tubes.

### Step 2: Select Tubes to Pick 🎯
- Look at the live video stream on the left side of the screen.
- Detected tubes have bright number badges numbered strictly from **left to right** (`#1`, `#2`, `#3`...).
- Click the tube buttons (`#1`, `#2`, `#3`...) in the exact order you want the robot to grasp them:
  - *Example:* Click `#3`, then `#1`. The robot will pick tube #3 first, then tube #1.
  - Selected tubes appear in the **PICKING QUEUE** box. If you made a mistake, click `✕` on any item or click `Clear All`.

### Step 3: Click "GRASP QUEUE" ▶
- Click the big green **"GRASP QUEUE"** button.
- The camera frame freezes to lock the 3D coordinates in place.
- The robot executes the grasp automatically:
  1. Hovers 10cm directly above the tube (**Pre-grasp hover**).
  2. Descends straight onto the tube.
  3. Clamps the gripper firmly.
  4. Lifts straight up by 10cm (**Retreat Waypoint 1**).
  5. Smoothly tilts away to safe transit height (**Retreat Waypoint 2**).
  6. Returns to the Ready Pose and opens the gripper.
- Once all selected tubes are grasped, the video unfreezes back to live mode!

---

## 3. Action Buttons Cheat Sheet

All buttons are on the right side of the main **Visual Grasping** screen:

| Button | Label | What it does | When to use |
|---|---|---|---|
| - | **GRASP QUEUE** | Freezes the frame and picks queued tubes one by one. | When you are ready to pick your selected tubes. |
| - | **EMERGENCY STOP** | Immediately stops any ongoing motion, opens gripper to Ready Open Width, returns arm safely to Home, and disables motors in strict order. | Any time during execution if a problem occurs or for emergency abort. |
| - | **Ready Pose** | Moves the arm to the default overhead scanning pose. | Always click this first before picking tubes. |
| - | **Enable Robot** | Activates motor holding torque without moving. | If motors were disabled, re-enable before motion. |
| - | **Disable Robot** | Moves arm to Safe Home first, then cuts motor torque (limp mode) for safety. | When powering off or moving the arm manually by hand. |
| - | **Safe Home** | Releases the gripper and smoothly parks arm at joint zero. | When finishing work, or to safely abort motion. |
| - | **Macro Center** | Centers the camera directly above the selected tube. | To inspect a tube directly beneath the camera. |
| - | **Reset / Unfreeze** | Cancels frozen frame and restores live camera view. | If you want to cancel a grasp or clear the view. |
| - | **Resume from Pause** | Resumes motion after a programmed waypoint pause. | Only appears if an active waypoint paused the arm. |

---

## 4. Top Status Badges & Hardware LED Indicator

Look at the top header bar and the action control panel to quickly check verified system status:

- `LIVE`: Camera is actively streaming and scanning the rack.
- `FROZEN`: Frame is locked while the robot is grasping to prevent coordinate drift.
- **Hardware Feedback LED Indicator** (in header & above action buttons):
  - `● Glowing Green LED (ROBOT ENABLED)`: **Real hardware feedback confirmed!** The robot driver verified via CAN bus communication that all joint motors are energized, holding torque is active, and the arm is holding pose.
  - `● Dim Gray LED (ROBOT DISABLED)`: Motor torque is released (limp mode, safe to move the arm by hand), or the hardware is unpowered/disconnected.
  - `● Pulsing Amber LED (VERIFYING FEEDBACK)`: The "Enable Robot" command was transmitted to the arm; the system is actively waiting for motor feedback frames before turning green. If hardware does not respond, the LED will remain dim gray and an error will be logged.
- `2 WPs (0 App / 2 Ret)`: Number of active motion waypoints (0 approach, 2 retreat waypoints). Click it to jump directly to the Waypoints Editor!
- `Connected / Hardware`: Indicates whether connected to physical arm or running in simulated dry-run mode.

---

## 5. Navigation Tabs Overview

The dashboard has 4 tabs at the top:

1. **Visual Grasping**:
   - The primary workstation screen.
   - Shows the live camera feed, tube selection buttons, picking queue, action buttons, and system telemetry console.
2. **Waypoints Editor**:
   - Allows viewing and editing robot via-points before grasping (**Approach**) and after grasping (**Retreat**).
   - Any changes are **automatically saved** to `config/waypoints.yaml` on disk.
   - By default, it uses the proven safe retreat path: lift +10cm vertically, then tilt-away pose.
3. **Robot & Camera Config**:
   - Configure camera focus, table height, pre-grasp clearances, **grasp Z offset/depth** (e.g. `-0.020m` to plunge deeper onto the tube body without shifting horizontal X/Y), ready pose coordinates, and **gripper width parameters** (Ready Open Width `0.055m`, Fixed Close Width `0.030m`, and Close Offset Margin `0.010m`).
   - Includes an instant **"Test Gripper Width Now"** tool to verify jaw motion on hardware.
   - Saves directly to `config/default.yaml` and synchronizes automatically on disk.
4. **User Guide**:
   - An on-screen version of this guide accessible at any time with one click!

---

## 6. 🚨 Troubleshooting & Safety FAQ

### Q: Why didn't the robot move when I started the server?
**A:** This is an intentional safety feature! On startup, the system powers on holding torque so the arm does not sag, but it **never moves automatically**. You must click **"Ready Pose"** when you are ready.

### Q: How do I stop or abort motion safely if there is a problem?
**A:** Click the red **"EMERGENCY STOP"** button on the station dashboard. No matter which step or trajectory the robot is currently performing, the system executes a strict 3-step safe abort sequence in exact order:
1. **STOP**: Immediately aborts the active motion and trajectory within 50ms, clearing remaining queued targets.
2. **RETURN HOME with gripper open**: Automatically commands the gripper open to Ready Open Width (`0.055m` / `5.5cm`) so any clamped object is released, then smoothly calculates and follows a minimum-jerk trajectory back to the zero safe home pose.
3. **DISABLE**: Cuts physical motor torque (`disable_all()`) once safely at home, putting motors in limp mode and updating the hardware feedback LED to dim gray.

### Q: What if an error occurs?
**A:** An alert banner (`RUNTIME ERROR DETECTED`) will appear above the System Telemetry console. Click **"View Stack Traceback"** to see what happened, or click `EMERGENCY STOP` or `Safe Home` and then restart the operation.

### Q: Can I pick tubes in reverse or random order?
**A:** Yes! Tubes are numbered #1 to #N from left to right. You can click `#4`, then `#2`, then `#1`—the robot will execute the exact order listed in your **PICKING QUEUE**.

### Q: How do I test the GUI without moving the physical robot?
**A:** Start the server with the `--dry-run` flag:
```bash
python3 scripts/web_grasp_gui.py --dry-run --port 8000
```
This runs the full vision and queue pipeline with a simulated robot.

### Q: What happens if I close the terminal or press Ctrl+C?
**A:** When `Ctrl+C` (or SIGINT/SIGTERM) is pressed in the terminal, the shutdown handler immediately cuts motor holding torque (`disable_all()`) and disconnects the CAN bus interface so the robot is safely disabled rather than left energized and holding torque.
