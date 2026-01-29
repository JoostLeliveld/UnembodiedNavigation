# Unembodied Navigation Repository Documentation

## Overview

**Unembodied Navigation** is a ROS 2 robotics project implementing autonomous navigation, path planning, control, and perception for a TurtleBot3 Burger robot. The project uses Gazebo Harmonic as the physics simulator and runs on Ubuntu 22.04 with ROS 2 Humble.

The repository is structured as a colcon workspace with multiple Python-based ROS 2 packages that work together to create a complete robot navigation stack. The architecture follows a modular design pattern where each package handles a specific aspect of robot functionality.

---

## Technology Stack

| Component | Version |
|-----------|---------|
| **Operating System** | Ubuntu 22.04 |
| **Middleware** | ROS 2 Humble |
| **Simulator** | Gazebo Harmonic |
| **Language** | Python 3.10 |
| **Build System** | colcon (ROS 2 build system) |
| **Package Format** | Python ament_python |

---

## Directory Structure

```
/home/joostleliveld/UnembodiedNavigation/
├── src/                        # Source packages (ROS 2 workspace source)
│   ├── control/                # Motor control & differential drive
│   ├── estimation/             # State estimation & odometry
│   ├── experiments/            # Experimental & testing code
│   ├── mapping/                # Occupancy grid & SLAM
│   ├── perception/             # Sensor processing & filtering
│   ├── planning/               # Path planning & navigation
│   ├── sim/                    # Gazebo simulation & robot description
│   └── visualization/          # RViz visualization
├── build/                      # Build artifacts (colcon build output)
├── install/                    # Installed packages & setup scripts
├── log/                        # Colcon build logs
└── PROJECT_STRUCTURE_SUMMARY.txt  # High-level project overview
```

---

## Core Packages

### 1. **SIM Package** (`src/sim/`)
**Purpose**: Gazebo simulation environment, robot description, and launcher configuration.

**Key Files & Components**:
- **Robot Description** (`robot_description/urdf/turtlebot3_burger.urdf.xacro`)
  - Complete URDF-XACRO model of TurtleBot3 Burger
  - Defines robot structure, collision geometry, and inertial properties
  - Includes base link, wheels (left/right), IMU, and LiDAR sensor definitions
  - Uses namespace support for multi-robot scenarios
  - Loads visual meshes from package://robot_description/meshes/

- **Launch Files**:
  - `sim.launch.py`: Main simulation launcher orchestrating all components
    - Launches Gazebo environment
    - Loads robot description
    - Spawns robot into simulation with 3-second delay
  - `gazebo.launch.py`: Gazebo environment initialization
    - Sets GZ_SIM_RESOURCE_PATH for mesh discovery
    - Loads empty world from `gazebo_worlds/worlds/empty/world.sdf`
    - Uses ros_gz_sim plugin for ROS 2 ↔ Gazebo communication
  - `robot_description.launch.py`: Robot model publisher
    - Executes xacro to generate URDF from XACRO
    - Launches `joint_state_publisher_gui` for manual joint control
    - Launches `robot_state_publisher` to broadcast TF frames

- **Gazebo Assets**:
  - `gazebo_worlds/worlds/empty/`: Baseline empty Gazebo world
  - `models/`: Custom Gazebo models (if any)

**Dependencies**: 
- `ros_gz_sim` (Gazebo-ROS 2 bridge)
- `joint_state_publisher_gui`
- `robot_state_publisher`
- `xacro`

**Current Status**: Fully functional simulation environment

---

### 2. **PLANNING Package** (`src/planning/`)
**Purpose**: Path planning and navigation algorithms.

**Key Files & Components**:
- `planning/astar_planner.py`: A* path planning node
  - Minimal ROS 2 node structure with logging
  - Extends `rclpy.node.Node`
  - Ready for expansion with actual A* algorithm and service/action interfaces
  - Currently placeholder for pathfinding logic

**Current Status**: Skeleton implementation, requires completion with actual A* algorithm

---

### 3. **CONTROL Package** (`src/control/`)
**Purpose**: Motor control, differential drive, and velocity command execution.

**Key Files & Components**:
- `control/__init__.py`: Empty Python module
- Basic package structure only
- No active nodes or implementations yet

**Intended Functionality**: 
- Convert velocity commands from planning layer to motor outputs
- Implement differential drive kinematics
- Manage motor PWM/velocity control

**Current Status**: Skeleton only, no implementation

---

### 4. **ESTIMATION Package** (`src/estimation/`)
**Purpose**: State estimation, odometry, and sensor fusion.

**Key Files & Components**:
- Basic package structure with `__init__.py`
- No active nodes

**Intended Functionality**:
- Fuse IMU and wheel encoder data for odometry
- Estimate robot pose and twist
- Implement EKF or Kalman filter for state estimation

**Current Status**: Skeleton only, no implementation

---

### 5. **PERCEPTION Package** (`src/perception/`)
**Purpose**: Sensor processing and perception pipeline.

**Key Files & Components**:
- Basic package structure
- No active nodes

**Intended Functionality**:
- Process LiDAR scan data
- Filter and transform sensor messages
- Implement obstacle detection
- Feed processed data to mapping and planning layers

**Current Status**: Skeleton only, no implementation

---

### 6. **MAPPING Package** (`src/mapping/`)
**Purpose**: Occupancy grid generation and SLAM (Simultaneous Localization and Mapping).

**Key Files & Components**:
- Basic package structure
- No active nodes

**Intended Functionality**:
- Build occupancy grids from LiDAR and perception data
- Implement or integrate SLAM algorithms
- Maintain and update environment maps for planning

**Current Status**: Skeleton only, no implementation

---

### 7. **EXPERIMENTS Package** (`src/experiments/`)
**Purpose**: Experimental code, testing scenarios, and research implementations.

**Key Files & Components**:
- Basic package structure with config and launch directories
- Reserved for experiment-specific implementations

**Intended Use**:
- Testing different planning or control algorithms
- Running benchmark scenarios
- Implementing variations of navigation approaches

**Current Status**: Empty skeleton, available for experimental work

---

### 8. **VISUALIZATION Package** (`src/visualization/`)
**Purpose**: RViz2 visualization configuration and launcher.

**Key Files & Components**:
- `launch/rviz.launch.py`: RViz2 launcher
  - Loads RViz configuration from `visualization/config/sim.rviz`
  - Enables `use_sim_time: True` for synchronization with Gazebo clock
  - Provides 3D visualization of robot state, TF frames, and sensor data
  - Allows visualization of planning results and trajectory

**Current Status**: Functional visualization framework

---

## Data Flow & Architecture

### System Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION LAYER                        │
│  Gazebo Harmonic (Physics, Actuators, Sensors)            │
│  ├─ TurtleBot3 Robot Model                                │
│  ├─ IMU Sensor                                            │
│  ├─ LiDAR Sensor                                          │
│  └─ Wheel Actuators (Differential Drive)                  │
└─────────────────────────────────────────────────────────────┘
           ▲                                 │
           │ (TF Frames, Sensor Data)       │ (Twist Commands)
           │                                 ▼
┌─────────────────────────────────────────────────────────────┐
│              PERCEPTION & STATE ESTIMATION                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PERCEPTION (perception/)                            │   │
│  │ - Process LiDAR scans                              │   │
│  │ - Filter sensor noise                              │   │
│  │ - Publish processed point clouds                   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ESTIMATION (estimation/)                            │   │
│  │ - Fuse IMU + wheel odometry                        │   │
│  │ - Estimate robot pose & velocity                   │   │
│  │ - Publish odometry & TF                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           ▲                                 │
           │ (Map, Pose)                    │ (Sensor Data)
           │                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 MAPPING & LOCALIZATION                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ MAPPING (mapping/)                                  │   │
│  │ - Build occupancy grids                            │   │
│  │ - Implement SLAM                                   │   │
│  │ - Maintain world model                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           ▲                                 │
           │ (Goal Pose)                    │ (Costmap, Map)
           │                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    PLANNING LAYER                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PLANNING (planning/)                                │   │
│  │ - A* path planning algorithm                       │   │
│  │ - Route calculation                                │   │
│  │ - Trajectory generation                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           ▲                                 │
           │ (Goal)                         │ (Planned Path)
           │                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL LAYER                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CONTROL (control/)                                  │   │
│  │ - Differential drive kinematics                    │   │
│  │ - Velocity control                                 │   │
│  │ - Motor command generation                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           ▲
           │ (Twist cmd_vel)
           │
        VISUALIZATION
  ┌─────────────────────────┐
  │ RViz2 (visualization/)  │
  │ - Show robot state      │
  │ - Display costmaps      │
  │ - Visualize paths       │
  │ - Monitor TF frames     │
  └─────────────────────────┘
```

### ROS 2 Topics & Services (Planned Architecture)

**Key Interfaces** (when fully implemented):

| Topic/Service | Type | Purpose |
|---------------|------|---------|
| `/cmd_vel` | Topic | Velocity commands (Twist) → Control |
| `/odom` | Topic | Odometry data ← Estimation |
| `/scan` | Topic | LiDAR scan data ← Gazebo |
| `/imu` | Topic | IMU measurements ← Gazebo |
| `/map` | Topic | Occupancy grid ← Mapping |
| `/plan` | Topic/Service | Path planning requests/responses |
| `/costmap` | Topic | Cost map for planning ← Mapping |
| `/tf` | Topic | TF frame broadcast ← Estimation/SIM |
| `/tf_static` | Topic | Static frame definitions |

---

## Launch Sequence & Startup Flow

### Starting the Simulation

```bash
# Terminal 1: Launch Gazebo simulator and robot
ros2 launch sim sim.launch.py

# Terminal 2: Launch RViz visualization
ros2 launch visualization rviz.launch.py
```

### Detailed Launch Sequence (from `sim.launch.py`)

1. **Gazebo Environment** (`gazebo.launch.py`)
   - Sets `GZ_SIM_RESOURCE_PATH` environment variable to find meshes
   - Launches Gazebo Harmonic with empty world file
   - Initializes physics engine and sensor simulation

2. **Robot Description** (`robot_description.launch.py`)
   - Executes xacro command to convert `turtlebot3_burger.urdf.xacro` → URDF
   - Launches `joint_state_publisher_gui` node
     - Allows manual manipulation of robot joints in GUI
     - Publishes joint states
   - Launches `robot_state_publisher` node
     - Reads robot description parameter
     - Subscribes to joint states
     - Publishes TF frames for entire robot kinematic tree

3. **Robot Spawn** (3-second delay, then `ros_gz_sim create`)
   - Waits 3 seconds for Gazebo to fully initialize
   - Uses `ros_gz_sim create` executable to spawn robot into Gazebo
   - Reads robot description from topic
   - Creates entity in Gazebo with initial position

### RViz Launch (`visualization/rviz.launch.py`)

- Loads configuration from `visualization/config/sim.rviz` (if exists)
- Enables simulation time synchronization with Gazebo
- Provides real-time 3D visualization

---

## Robot Model & Configuration

### TurtleBot3 Burger Physical Specification

**From `robot_description/urdf/turtlebot3_burger.urdf.xacro`**:

| Component | Details |
|-----------|---------|
| **Base Link** | Mass: 0.826 kg, Collision box: 0.14×0.14×0.143 m |
| **Wheels** | Type: Continuous joint, Radius: 33 mm, Width: 18 mm, Mass: 0.028 kg each |
| **Wheelbase** | 160 mm (80 mm distance from center) |
| **Sensors** | IMU, LiDAR (lds.stl model) |
| **Coordinate Frames** | base_footprint, base_link, wheel_left, wheel_right, imu_link, base_scan |

**Mesh Models**:
- Base: `burger_base.stl`
- Wheels: `left_tire.stl`, `right_tire.stl`
- LiDAR: `lds.stl`
- All meshes sourced from `robot_description/meshes/turtlebot3_description/`

---

## Build & Installation

### Build System

- **Build Tool**: `colcon` (ROS 2 standard)
- **Package Type**: Python ament_python
- **Build Output**: 
  - `build/`: Intermediate build files
  - `install/`: Final installed packages with Python site-packages
  - `log/`: Colcon build logs

### Build Artifacts Location

```
install/
├── control/lib/python3.10/site-packages/control/
├── estimation/lib/python3.10/site-packages/estimation/
├── experiments/lib/python3.10/site-packages/experiments/
├── mapping/lib/python3.10/site-packages/mapping/
├── perception/lib/python3.10/site-packages/perception/
├── planning/lib/python3.10/site-packages/planning/
├── sim/lib/python3.10/site-packages/sim/
├── visualization/lib/python3.10/site-packages/visualization/
└── [Setup scripts for sourcing]
```

### Sourcing the Workspace

```bash
# Setup ROS 2 and workspace
source /opt/ros/humble/setup.bash
source /home/joostleliveld/UnembodiedNavigation/install/local_setup.bash
```

---

## File Relationships & Dependencies

### Dependency Graph

```
SIM Package (Gazebo Simulator)
    ├── robot_description/urdf/ → Defines robot geometry
    ├── gazebo_worlds/ → Environment definition
    ├── Depends on: ros_gz_sim, joint_state_publisher_gui, robot_state_publisher
    └── Used by: All other packages for sensor/actuator communication

VISUALIZATION Package (RViz)
    ├── Requires: sim package (for robot description)
    ├── Displays: TF frames, sensor data, planned paths
    └── No runtime dependencies on control/planning/estimation

PLANNING Package (A* Planner)
    ├── Will consume: costmap from MAPPING
    ├── Will consume: current pose from ESTIMATION
    ├── Produces: planned path trajectory
    └── Consumed by: CONTROL

CONTROL Package (Motor Control)
    ├── Consumes: planned trajectory from PLANNING
    ├── Consumes: odometry from ESTIMATION
    ├── Produces: cmd_vel (Twist) to robot
    └── Depends on: differential drive kinematics

ESTIMATION Package (State Estimation)
    ├── Consumes: /scan from SIM (LiDAR)
    ├── Consumes: /imu from SIM
    ├── Consumes: wheel encoder feedback (from CONTROL)
    ├── Produces: /odom (Odometry), /tf (transforms)
    └── Used by: PLANNING, CONTROL, VISUALIZATION

MAPPING Package (SLAM/Occupancy Grid)
    ├── Consumes: /scan from SIM
    ├── Consumes: odometry/pose from ESTIMATION
    ├── Produces: /map, /costmap
    └── Used by: PLANNING

PERCEPTION Package (Sensor Processing)
    ├── Consumes: raw sensor data from SIM
    ├── Produces: filtered/processed sensor data
    └── Feeds: MAPPING, ESTIMATION, possibly PLANNING
```

---

## Current Implementation Status

| Package | Status | Completeness |
|---------|--------|--------------|
| **SIM** | ✅ Functional | 100% - Full Gazebo integration |
| **VISUALIZATION** | ✅ Functional | 100% - RViz launcher ready |
| **PLANNING** | ⚠️ Partial | 5% - Node skeleton only, A* algorithm needed |
| **CONTROL** | ❌ Not Started | 0% - No implementation |
| **ESTIMATION** | ❌ Not Started | 0% - No implementation |
| **MAPPING** | ❌ Not Started | 0% - No implementation |
| **PERCEPTION** | ❌ Not Started | 0% - No implementation |
| **EXPERIMENTS** | ❌ Not Started | 0% - Empty skeleton |

### Implementation Priorities

1. **ESTIMATION** - Required first; provides core robot state (pose, velocity)
2. **PERCEPTION** - Process LiDAR and sensor data
3. **MAPPING** - Build world model from perception
4. **PLANNING** - Use map to calculate paths (A* already has skeleton)
5. **CONTROL** - Execute planned paths with motor control
6. **EXPERIMENTS** - Run scenarios and test implementations

---

## How Each File Works & Relates

### Entry Points & Launchers

- **`src/sim/launch/sim.launch.py`**: Master orchestrator
  - Calls `gazebo.launch.py` → Starts physics engine
  - Calls `robot_description.launch.py` → Publishes robot model & TF
  - Spawns robot into Gazebo after 3-second delay
  - **Used by**: Direct user launch, other packages depend on its outputs

- **`src/visualization/launch/rviz.launch.py`**: Visualization entry
  - Depends on: working SIM + ESTIMATION (for TF)
  - Displays: robot model, sensor data, planning results
  - **Used by**: User for debugging and visualization

### Node Implementations

- **`src/planning/planning/astar_planner.py`**: 
  - Node class extending `rclpy.node.Node`
  - Currently logs startup message only
  - Will implement path planning algorithm
  - **Relationships**: Will subscribe to `/costmap` and `/scan`, serve planning requests

### URDF/Xacro Definitions

- **`src/sim/robot_description/urdf/turtlebot3_burger.urdf.xacro`**:
  - Generates complete robot kinematic tree
  - Referenced by: `robot_description.launch.py` via xacro command
  - Consumed by: `robot_state_publisher` (publishes TF)
  - Loaded into: Gazebo during spawn (read from `/robot_description` topic)
  - Used by: RViz for visualization

### Configuration Files

- **`visualization/config/sim.rviz`**: RViz configuration
  - Specifies which topics to visualize
  - Display setup for robot, sensors, trajectories
  - Referenced by: `rviz.launch.py`

- **`src/sim/gazebo_worlds/worlds/empty/world.sdf`**: Gazebo world definition
  - Defines environment (ground plane, lighting, physics)
  - Referenced by: `gazebo.launch.py`
  - Consumed by: Gazebo during initialization

---

## Testing & Validation

### Current Working Features

1. **Gazebo Simulation**: Fully functional
   - Robot spawning verified (seen in Gazebo)
   - Physics simulation active
   - Sensor simulation (IMU, LiDAR) operational

2. **TF Frame Broadcasting**: Working
   - Verified with `ros2 run tf2_tools view_frames`
   - Complete kinematic tree available

3. **RViz Visualization**: Functional
   - Real-time robot visualization
   - TF frame display
   - Sensor data visualization (with proper nodes)

### Testing Commands

```bash
# Verify simulation
ros2 launch sim sim.launch.py

# View TF tree
ros2 run tf2_tools view_frames

# Visualize in RViz
ros2 launch visualization rviz.launch.py

# Check active topics
ros2 topic list

# Inspect specific topics
ros2 topic echo /tf
ros2 topic echo /joint_states
```

---

## Missing Components & Future Work

### High-Priority Implementations Needed

1. **State Estimation Node** (estimation/)
   - Subscribe to LiDAR, IMU
   - Publish odometry & TF transforms
   - Implement sensor fusion (Kalman filter)

2. **Perception Pipeline** (perception/)
   - LiDAR point cloud processing
   - Obstacle detection
   - Sensor data filtering

3. **Mapping & SLAM** (mapping/)
   - Occupancy grid generation
   - SLAM algorithm
   - Cost map for planning

4. **Complete A* Planner** (planning/)
   - Implement actual A* algorithm
   - Service/action interface for planning requests
   - Path output format

5. **Control Layer** (control/)
   - Differential drive kinematics
   - Velocity command execution
   - Motor PID control

### Configuration Files Needed

- `visualization/config/sim.rviz`: RViz visualization config
- Config files in `control/config/`, `mapping/config/`, etc.

---

## Package Dependencies Summary

### External ROS 2 Packages Required

| Package | Purpose |
|---------|---------|
| `ros_gz_sim` | Gazebo-ROS 2 bridge |
| `robot_state_publisher` | Broadcast TF frames |
| `joint_state_publisher_gui` | Joint control GUI |
| `rviz2` | Visualization |
| `gazebo` (Harmonic) | Physics simulation |
| `xacro` | URDF processing |

### Python Dependencies

- `rclpy`: ROS 2 Python client library
- `std_msgs`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`: ROS message types
- Standard library: `os`, `logging`, etc.

---

## Summary Table: File Purposes & Relationships

| File | Purpose | Depends On | Used By | Status |
|------|---------|-----------|---------|--------|
| `sim.launch.py` | Main launcher | gazebo.launch.py, robot_description.launch.py | Direct user invocation | ✅ |
| `gazebo.launch.py` | Gazebo init | gazebo_worlds/worlds/empty/world.sdf | sim.launch.py | ✅ |
| `robot_description.launch.py` | Robot publisher | turtlebot3_burger.urdf.xacro | sim.launch.py | ✅ |
| `turtlebot3_burger.urdf.xacro` | Robot model | Mesh files in robot_description/meshes/ | robot_description.launch.py | ✅ |
| `rviz.launch.py` | RViz launcher | sim.rviz config (if exists) | Direct user invocation | ✅ |
| `astar_planner.py` | Path planner | (mapping, estimation when complete) | control | ⚠️ |
| `control/` | Motor control | astar_planner.py, estimation outputs | Robot actuators | ❌ |
| `estimation/` | State estimation | Gazebo sensor outputs | control, planning, mapping | ❌ |
| `mapping/` | Environment model | perception, estimation outputs | planning | ❌ |
| `perception/` | Sensor processing | Gazebo sensor outputs | mapping, estimation | ❌ |

---

## Quick Start Guide

### Prerequisites
```bash
# Ensure ROS 2 Humble is installed
source /opt/ros/humble/setup.bash
```

### Build Workspace
```bash
cd /home/joostleliveld/UnembodiedNavigation
colcon build --symlink-install
source install/local_setup.bash
```

### Run Simulation
```bash
# Terminal 1: Start Gazebo + Robot
ros2 launch sim sim.launch.py

# Terminal 2: Visualize in RViz
ros2 launch visualization rviz.launch.py
```

### Verify System
```bash
# Check TF frames
ros2 run tf2_tools view_frames

# Monitor topics
ros2 topic list
ros2 topic echo /tf
```

---

## Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNEMBODIED NAVIGATION                        │
│                    ROS 2 + Gazebo Stack                         │
└─────────────────────────────────────────────────────────────────┘

HARDWARE SIMULATION LAYER
┌─────────────────────────────────────────────────────────────────┐
│ Gazebo Harmonic (Sim Package)                                  │
│ ├─ TurtleBot3 Burger URDF Model                                │
│ ├─ Physics Engine (ODE)                                        │
│ ├─ IMU Sensor Simulation                                       │
│ ├─ LiDAR Sensor Simulation                                     │
│ └─ Wheel Actuators (Differential Drive)                        │
└─────────────────────────────────────────────────────────────────┘

PERCEPTION & FUSION LAYER
┌────────────────────────────┬────────────────────────────────────┐
│ Perception (perception/)   │ Estimation (estimation/)           │
│ - LiDAR Processing        │ - Odometry Calculation            │
│ - Point Cloud Filtering   │ - Sensor Fusion (EKF/UKF)        │
│ - Obstacle Detection      │ - Pose Estimation                 │
└────────────────────────────┴────────────────────────────────────┘

MAPPING & LOCALIZATION LAYER
┌─────────────────────────────────────────────────────────────────┐
│ Mapping (mapping/)                                              │
│ ├─ SLAM Algorithm                                              │
│ ├─ Occupancy Grid Generation                                   │
│ └─ Global Cost Map Creation                                    │
└─────────────────────────────────────────────────────────────────┘

PLANNING & DECISION LAYER
┌─────────────────────────────────────────────────────────────────┐
│ Planning (planning/)                                            │
│ ├─ A* Path Planning Algorithm                                  │
│ ├─ Trajectory Generation                                       │
│ └─ Goal & Navigation Management                                │
└─────────────────────────────────────────────────────────────────┘

EXECUTION LAYER
┌─────────────────────────────────────────────────────────────────┐
│ Control (control/)                                              │
│ ├─ Differential Drive Kinematics                              │
│ ├─ Velocity Controller                                         │
│ └─ Motor Command Generation (Twist → Wheel Velocities)        │
└─────────────────────────────────────────────────────────────────┘

VISUALIZATION LAYER
┌─────────────────────────────────────────────────────────────────┐
│ RViz2 (visualization/)                                          │
│ ├─ Robot Model Display                                         │
│ ├─ Sensor Data Visualization                                  │
│ ├─ Path & Trajectory Display                                  │
│ └─ TF Frame Broadcasting                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts & ROS 2 Patterns

### Publish-Subscribe Architecture
Each component publishes its outputs as ROS 2 topics and subscribes to required inputs. This loose coupling allows packages to be developed and tested independently.

### Transformation Broadcasting (TF)
The `robot_state_publisher` maintains the kinematic tree via TF2. All coordinate transformations (base_link → wheel, base_link → scan, etc.) are published to `/tf`, enabling all other components to work in the correct coordinate frames.

### Launch System
ROS 2 Python launch files (`*.launch.py`) orchestrate node startup, parameter passing, and inter-node dependencies. The modular launch structure allows independent testing of subsystems.

### Simulation Time
When using Gazebo, all ROS 2 nodes use simulation time (`/clock` topic) rather than wall time. This synchronizes all components perfectly and allows running simulations at different speeds.

---

**Document Generated**: January 29, 2026  
**ROS 2 Distribution**: Humble  
**Gazebo Version**: Harmonic  
**Ubuntu Version**: 22.04 LTS
