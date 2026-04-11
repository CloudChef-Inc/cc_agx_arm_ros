# cc_agx_arm_ros — dual-arm Nero fork

Fork of [`agilexrobotics/agx_arm_ros`](https://github.com/agilexrobotics/agx_arm_ros) that adds a dual-arm setup for two **AgileX Nero 7-DoF arms** side-mounted as left/right shoulders on a torso, plus a tiny browser webapp for per-joint control.

The upstream single-arm driver, URDFs, and MoveIt configs are preserved untouched. The upstream English README is available at [`README_EN.md`](README_EN.md).

## What this fork adds

Three new packages under `src/`:

| Package | Purpose |
| --- | --- |
| `nero_dual_description` | Composed two-arm URDF. Wraps the upstream Nero xacro in a prefixed `<xacro:macro>` and instantiates it twice on a torso. |
| `nero_webapp` | FastAPI + WebSocket ROS 2 node. Serves `http://<robot>:8080` with per-joint sliders, a stick-figure 3D view, and Send buttons that publish `sensor_msgs/JointState` to the drivers. |
| `nero_dual_bringup` | One-shot launch file: `robot_state_publisher` + both arm drivers namespaced under `/left` and `/right` + the webapp. |

## Hardware assumptions

- Two AgileX Nero 7-DoF arms on CAN (defaults: `can0` = left, `can1` = right).
- Side-mounted as shoulders on a torso. Mount geometry is parameterised at the top of `src/nero_dual_description/urdf/two_nero.urdf.xacro` — tune the offsets there to match your physical rig.
- Pika grippers are installed but **not yet controlled**: `agilexrobotics/pika_ros` is ROS 1 Noetic only, so `effector_type:=none` is used. When a ROS 2 driver lands, wire it into `dual_arm.launch.py`.
- Target platform: Ubuntu 24.04 + ROS 2 Jazzy.

## First-time setup (on the robot machine)

```bash
git clone --recurse-submodules git@github.com:aloor12/cc_agx_arm_ros.git ~/cc-nero
cd ~/cc-nero

# Dependencies (FastAPI/Uvicorn are not in rosdep on Jazzy yet).
sudo apt install python3-fastapi python3-uvicorn
rosdep install --from-paths src --ignore-src -r -y || true

colcon build --symlink-install
source install/setup.bash
```

## Running

```bash
# 1. Bring CAN up.
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000

# 2. Launch drivers + webapp.
source install/setup.bash
ros2 launch nero_dual_bringup dual_arm.launch.py
```

Open `http://<robot-host>:8080` from any LAN browser (or `http://localhost:8080` on the robot itself).

### Override CAN ports or HTTP port

```bash
ros2 launch nero_dual_bringup dual_arm.launch.py \
    left_can:=can0 right_can:=can1 http_port:=8080
```

### URDF sanity-check without hardware

```bash
ros2 launch nero_dual_description display.launch.py
```

Shows both arms in RViz with a `joint_state_publisher_gui` so you can sweep every joint through its URDF limits before energising the motors.

## Topics

The upstream driver is launched twice under `/left` and `/right` namespaces:

| Direction | Topic | Type |
| --- | --- | --- |
| command | `/left/control/joint_states`, `/right/control/joint_states` | `sensor_msgs/JointState` |
| feedback | `/left/feedback/joint_states`, `/right/feedback/joint_states` | `sensor_msgs/JointState` |

The webapp node publishes commands and subscribes to feedback on those exact topics. Commanded positions are clamped server-side to the URDF joint limits from `nero_description.urdf` before publishing.

## Staying in sync with upstream

```bash
# one-time
git remote add upstream https://github.com/agilexrobotics/agx_arm_ros.git

# periodically
git fetch upstream
git merge upstream/ros2          # or whatever upstream branch you track
git submodule update --init --recursive
```

Conflicts should generally only appear on this `README.md`.

## Known limitations

- Per-joint control only — no IK, no MoveIt, no trajectories from the UI.
- The browser 3D view is a stick figure, not a mesh render. The URDF feeds `robot_state_publisher` for ROS-side TF correctness.
- Pika gripper control is stubbed (no ROS 2 driver upstream).
