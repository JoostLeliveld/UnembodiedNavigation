# Codebase Architecture Overview: Package & Component Organization
## Structural Relationships in core_tue4tm00_humble

**Focus:** How packages, worlds, launch files, URDFs, robots, RViz, control, and sensors are organized and related

**Date:** January 29, 2026

---

## Overview: Component Layers & Relationships

The codebase is organized into **vertically integrated stacks** where each layer depends on lower layers:

```
┌─────────────────────────────────────────────────────────────┐
│  EXPERIMENT / APPLICATION LAYER                             │
│  ├─ Lecture Packages (core_tue4tm00_lectures_humble)        │
│  └─ Assignment Packages (core_tue4tm00_assignment*)         │
├─────────────────────────────────────────────────────────────┤
│  ROBOT INTERFACE LAYER                                      │
│  ├─ core_robot_control_humble (control algorithms)         │
│  ├─ core_robot_planning_humble (planning algorithms)       │
│  └─ core_tue4tm00_robot_humble (robot-specific wrappers)   │
├─────────────────────────────────────────────────────────────┤
│  SIMULATION LAYER                                           │
│  ├─ core_robot_simulate (robot spawning, sensors)          │
│  ├─ core_gazebo_worlds (world definitions)                 │
│  └─ core_gazebo_teleport (Gazebo utilities)                │
├─────────────────────────────────────────────────────────────┤
│  FOUNDATION LAYER                                           │
│  ├─ core_rviz_tools (visualization configs)                │
│  ├─ core_tf_tools (transform management)                   │
│  ├─ core_geometry_tools (math utilities)                   │
│  └─ core_launch_tools (launch helpers)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Organization Pattern

### Standard Package Structure

Every ROS 2 package follows this pattern:

```
package_name/
├── launch/              ← Launch files (orchestration)
│   ├── demo_*.launch.py (executable demonstrations)
│   └── *.launch.py      (reusable launchers)
│
├── config/              ← Configuration files
│   ├── *.rviz          (RViz configurations)
│   ├── *.yaml          (node parameters)
│   └── *.world         (Gazebo worlds, if applicable)
│
├── src/                 ← Implementation
│   ├── *.py            (Python nodes)
│   └── *.cpp           (C++ nodes)
│
├── models/              ← Robot/object models (simulation packages only)
│   └── robot_name/
│       ├── model.sdf.xacro    (Gazebo model definition)
│       ├── model.config       (Gazebo config)
│       └── meshes/            (3D mesh files)
│
├── worlds/              ← Gazebo worlds (simulation packages only)
│   └── *.world         (World definitions)
│
├── package.xml          ← ROS package metadata
├── CMakeLists.txt       ← Build configuration
└── README.md            ← Documentation
```

---

## Component 1: SIMULATION (core_robot_simulation_humble)

### Directory Structure & Files

```
core_robot_simulation_humble/
│
├── core_robot_simulate/
│   ├── launch/
│   │   ├── gazebo_world.launch.py          ← Starts Gazebo with world
│   │   ├── robocyl.launch.py               ← Spawns robot + sensors
│   │   ├── robocyl_gazebo_world.launch.py  ← Full simulation (combined)
│   │   └── rviz.launch.py                  ← RViz launcher
│   │
│   ├── config/
│   │   ├── gazebo_default.config           ← Gazebo client settings
│   │   └── robocyl.rviz                    ← Default RViz config
│   │
│   ├── models/
│   │   ├── robocyl/                        ← Main robot model
│   │   │   ├── model.sdf.xacro            ← Parameterized robot definition
│   │   │   ├── model.config               ← Gazebo model config
│   │   │   └── meshes/                    ← Visual meshes
│   │   ├── robobox/                       ← Alternative robot
│   │   └── robocyl_cylinder/              ← Another variant
│   │
│   ├── worlds/
│   │   ├── empty.world                    ← Minimal empty world
│   │   ├── core_office.world              ← Complex office environment
│   │   ├── warehouse.world                ← Industrial scenario
│   │   └── ... other scenarios ...
│   │
│   └── src/                               ← Gazebo plugins, if needed
│
├── core_gazebo_worlds/
│   └── models/
│       └── [shared environment models]    ← Walls, obstacles, furniture
│
└── core_gazebo_teleport/
    └── src/
        └── teleport_service.py            ← Gazebo entity teleportation
```

### How Simulation Components Relate

```
robocyl_gazebo_world.launch.py
├─ Includes: gazebo_world.launch.py
│   └─ Spawns: Gazebo process with empty.world / core_office.world
│
├─ Includes: robocyl.launch.py
│   ├─ Loads: model.sdf.xacro (RoboCyl robot definition)
│   ├─ Spawns: Robot in Gazebo at (x, y, z, yaw)
│   ├─ Bridges: Gazebo topics ↔ ROS 2 topics
│   │   └─ /model/robocyl/cmd_vel → /robocyl/cmd_vel
│   │   └─ /model/robocyl/odom → /robocyl/odom
│   │   └─ /model/robocyl/scan → /robocyl/scan
│   │
│   └─ Launches: robot_state_publisher (for TF/URDF)
│
└─ Includes: rviz.launch.py
    └─ Loads: robocyl.rviz config
```

### Key Files: Worlds & Models

**World File Structure** (`empty.world`):
```xml
<sdf version="1.6">
  <world name="empty">
    <physics type="ode">...</physics>  ← Physics engine
    <plugin ...>...</plugin>           ← Gazebo plugins
    <model name="ground_plane">...</model>  ← Static environment
    <light ...>...</light>             ← Lighting
  </world>
</sdf>
```

**Robot Model Structure** (`model.sdf.xacro`):
```xml
<sdf>
  <model name="${name}">              ← Parameterized name
    <link name="base_link">           ← Main chassis
      <visual>...</visual>
      <collision>...</collision>
      <inertial>...</inertial>
    </link>
    
    <link name="laser_frame">         ← Sensor frame
      <sensor name="laser">
        <ray>
          <range>
            <max>20.0</max>           ← Sensor specs
          </range>
        </ray>
      </sensor>
    </link>
    
    <joint name="base_laser">         ← Kinematic chain
      <parent>base_link</parent>
      <child>laser_frame</child>
    </joint>
  </model>
</sdf>
```

---

## Component 2: SENSORS & ROS-GAZEBO BRIDGE

### Sensor Types & Topic Flow

```
Gazebo Simulation
│
├─ Physics Engine
│  └─ Updates robot pose
│
├─ Laser Sensor (in model.sdf.xacro)
│  └─ Publishes: /model/robocyl/scan (Gazebo-internal)
│     └─ Bridge converts to: /robocyl/scan (ROS 2)
│
├─ Odometry (built-in or plugin)
│  └─ Publishes: /model/robocyl/odometry (Gazebo-internal)
│     └─ Bridge converts to: /robocyl/odom (ROS 2)
│
├─ IMU (optional in xacro)
│  └─ Publishes: /model/robocyl/imu (Gazebo-internal)
│     └─ Bridge converts to: /robocyl/imu (ROS 2)
│
└─ Camera (optional in xacro)
   └─ Publishes: /model/robocyl/camera (Gazebo-internal)
      └─ Bridge converts to: /robocyl/image_raw (ROS 2)
```

### Bridge Configuration (in robocyl.launch.py)

```python
ros_gz_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        # Gazebo→ROS (publishers) use [
        '/model/{namespace}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/model/{namespace}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        
        # ROS→Gazebo (subscribers) use ]
        '/model/{namespace}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
    ],
    remappings=[
        # Map internal Gazebo topics to ROS namespace
        ('/model/{namespace}/scan', '/{namespace}/scan'),
        ('/model/{namespace}/odom', '/{namespace}/odom'),
        ('/model/{namespace}/cmd_vel', '/{namespace}/cmd_vel'),
    ],
)
```

---

## Component 3: CONTROL (core_robot_control_humble)

### Control Package Organization

```
core_robot_control_humble/
│
├── core_unicycle_control/
│   ├── launch/
│   │   └── demo_unicycle_control.launch.py
│   ├── config/
│   │   └── controller_params.yaml          ← Gains, thresholds
│   └── src/
│       └── unicycle_control_node.py        ← Subscribes: /cmd_vel
│                                            ← Publishes: motor commands
│
├── core_path_follow_ctrl/
│   ├── launch/
│   │   └── demo_path_follower.launch.py
│   ├── config/
│   │   ├── path_follow_params.yaml
│   │   └── controller.rviz                 ← Path visualization
│   └── src/
│       └── path_follower_node.py           ← Subscribes: /path, /odom
│                                            ← Publishes: /cmd_vel
│
├── core_twist_tools/
│   └── src/
│       ├── teleop_key.py                   ← Keyboard → /cmd_vel
│       ├── teleop_joy.py                   ← Joystick → /cmd_vel
│       └── twist_mux.py                    ← Arbitrate multiple /cmd_vel sources
│
├── core_pose_tools/
│   └── src/
│       └── pose_teleop.py                  ← Keyboard → direct pose setting
│
└── core_scan_gap_follower/
    └── src/
        └── gap_follower.py                 ← Laser scan → reactive /cmd_vel
```

### Control Data Flow

```
Multiple Input Sources
├─ Keyboard (twist_teleop_key)
├─ Joystick (twist_teleop_joy)
├─ Path Planner → /path
│   └─ path_follower_node
│       └─ Outputs: /cmd_vel
│
├─ Goal (from RViz "Set Goal")
│   └─ goal_nav_controller
│       └─ Outputs: /cmd_vel
│
└─ Laser Scan → /scan
   └─ scan_gap_follower
       └─ Outputs: /cmd_vel

All sources → twist_mux.py (Multiplexer)
              └─ Selects highest-priority source
                 └─ Outputs final: /cmd_vel

/cmd_vel → Robot (via ROS-Gazebo bridge)
           └─ Gazebo applies command to physics
              └─ Sensor data generated
                 └─ Published back to ROS (/scan, /odom)
```

### Config Files: What They Contain

**controller_params.yaml** (example):
```yaml
unicycle_controller:
  linear_velocity_max: 0.5        # m/s
  angular_velocity_max: 1.0       # rad/s
  acceleration_max: 0.2
  deceleration_max: 0.3
  wheel_base: 0.1                 # m (distance between wheels)
  update_rate: 10                 # Hz

path_follower:
  lookahead_distance: 0.5         # m
  max_steering_angle: 1.57        # rad (90 degrees)
  kp: 1.0                         # Proportional gain
  ki: 0.1
  kd: 0.05
```

---

## Component 4: PLANNING (core_robot_planning_humble)

### Planning Package Organization

```
core_robot_planning_humble/
│
├── core_occupancy_grid_costmap/
│   ├── launch/
│   │   └── demo_costmap.launch.py
│   ├── config/
│   │   ├── occupancy_params.yaml
│   │   └── costmap.rviz
│   └── src/
│       └── occupancy_mapper.py             ← Subscribes: /scan
│                                            ← Publishes: /occupancy_grid
│
├── core_search_path_planner/
│   ├── launch/
│   │   └── demo_search_planner.launch.py
│   ├── config/
│   │   ├── planner_params.yaml
│   │   └── planner.rviz
│   └── src/
│       └── astar_planner.py                ← Subscribes: /occupancy_grid, /goal_pose
│                                            ← Publishes: /path
│
├── core_sampling_path_planner/
│   ├── src/
│   │   └── rrt_planner.py                  ← RRT/RRT* implementation
│
├── core_path_tools/
│   └── src/
│       ├── path_smoother.py
│       └── path_simplifier.py
│
└── core_map_tools/
    └── src/
        └── map_server.py                   ← Loads YAML map files
```

### Planning Data Flow

```
Sensor Data (LaserScan)
    /scan
    ↓
occupancy_mapper.py
    └─ Integrates scan data over time
       └─ /occupancy_grid (OccupancyGrid msg)
          ↓
costmap.py
    └─ Inflates obstacles by robot radius
       └─ /costmap (OccupancyGrid msg)
          ↓
search_path_planner.py (A* or Dijkstra)
    └─ Also subscribes: /goal_pose (from RViz tool)
       └─ /path (Path msg: sequence of waypoints)
          ↓
path_follower.py (controller)
    └─ Subscribes: /path and /odom
       └─ /cmd_vel (Twist msg)
          ↓
Robot moves, generates new sensor data
    └─ Back to occupancy_mapper (feedback loop)
```

---

## Component 5: RVIZ CONFIGURATION

### RViz Config Organization

```
core_rviz_tools/
├── config/
│   ├── default.rviz                    ← Minimal (grid only)
│   ├── robot_scan.rviz                 ← Robot model + laser scan
│   ├── robot_scan_map.rviz             ← + occupancy grid map
│   ├── robot_scan_map_costmap.rviz     ← + inflated costmap
│   ├── robot_scan_map_costmap_path.rviz ← + planned path (full)
│   └── ... 12 more specialized configs
│
└── launch/
    └── rviz.launch.py                  ← Generic RViz launcher
        └─ Arguments:
            ├─ config: which .rviz file
            ├─ namespace: robot namespace
            ├─ fixed_frame: global frame (map/world)
            └─ use_sim_time: sync with Gazebo clock
```

### RViz Config Contents (YAML format)

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
    Expanded:
      - /Global Options1
      - /RobotModel1
      - /LaserScan1

Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid         # Grid background
    - Class: rviz_default_plugins/RobotModel   # URDF/SDF visualization
      Description Source: Topic
      Description Topic: /robot_description
    
    - Class: rviz_default_plugins/LaserScan    # Sensor data
      Topic: /robocyl/scan
      Color Transformer: Intensity
    
    - Class: rviz_default_plugins/Map          # Occupancy grid
      Topic: /occupancy_grid
      Color Scheme: costmap
    
    - Class: rviz_default_plugins/Path         # Planned path
      Topic: /path
      Color: 0; 255; 0
      Line Width: 0.05
    
    - Class: rviz_default_plugins/TF           # Transform frames
      Enabled: true
    
  Global Options:
    Fixed Frame: world (or map)
    Frame Rate: 30
    Background Color: 48; 48; 48
  
  Tools:
    - Class: rviz_default_plugins/SetGoal      # "Set 2D Goal" button
      Topic: /goal_pose
    - Class: rviz_default_plugins/SetInitialPose
      Topic: /initialpose
```

### RViz Launch Integration

```python
# In robocyl_gazebo_world.launch.py
rviz_launch = GroupAction(
    actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                'core_rviz_tools/launch/rviz.launch.py'
            ),
        )
    ],
    launch_configurations={
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        'config': 'robot_scan_map_costmap_path.rviz',  # Which config to load
        'fixed_frame': 'world',                         # Coordinate reference
        'namespace': LaunchConfiguration('namespace'),
    },
)
```

---

## Component 6: URDF & ROBOT STATE PUBLISHER

### URDF Role in the System

```
Gazebo Model (model.sdf)
└─ Contains kinematic structure:
   ├─ base_link (main chassis)
   ├─ laser_frame (sensor mounting point)
   ├─ wheel_left, wheel_right
   └─ Joints connecting them

Robot State Publisher Node
├─ Subscribes: /joint_states (from Gazebo)
├─ Has: robot_description (URDF from model.sdf)
└─ Publishes: /tf (transform frames)
   └─ world → map → odom → base_link → laser_frame
      └─ Used by RViz to:
         ├─ Draw robot correctly
         ├─ Transform sensor data
         └─ Coordinate frames for planning
```

**In launch file:**
```python
robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[
        {
            'robot_description': model_description,  # SDF/URDF from xacro
            'use_sim_time': True,                     # Sync with Gazebo
        }
    ],
    remappings=[
        ('/tf', 'tf'),                # Publish to global /tf
        ('/tf_static', 'tf_static'),
    ],
)
```

---

## Your Thesis Repository Structure (tue_ai_es_thesis_ws)

Based on the core_tue4tm00_humble patterns, your repo should look like:

```
tue_ai_es_thesis_ws/src/
│
├── sim/                              ← Simulation layer
│   ├── launch/
│   │   ├── gazebo.launch.py          ← Gazebo process
│   │   ├── turtlebot3.launch.py      ← Robot + bridge + sensors
│   │   └── full_simulation.launch.py ← Complete system
│   │
│   ├── config/
│   │   ├── gazebo_default.config     ← Gazebo GUI settings
│   │   └── empty.world               ← or warehouse.world
│   │
│   ├── worlds/
│   │   ├── empty.world
│   │   ├── warehouse.world
│   │   └── custom_scenario.world
│   │
│   └── package.xml
│
├── visualization/                   ← RViz & visualization
│   ├── launch/
│   │   ├── rviz.launch.py           ← Generic launcher
│   │   └── visualization.launch.py  ← Full pipeline viz
│   │
│   ├── config/
│   │   ├── perception.rviz
│   │   ├── mapping.rviz
│   │   ├── planning.rviz
│   │   ├── control.rviz
│   │   └── full_pipeline.rviz
│   │
│   └── package.xml
│
├── perception/                      ← Your perception layer
│   ├── launch/
│   │   └── perception.launch.py
│   ├── src/
│   │   ├── vision_pose_node.py      ← Subscribes: /turtlebot/scan
│   │   └── obstacle_detector.py
│   └── config/
│       └── perception_params.yaml
│
├── estimation/                      ← Your estimation layer
│   ├── launch/
│   │   └── estimator.launch.py
│   ├── src/
│   │   └── state_estimator.py       ← Subscribes: /turtlebot/odom
│   └── config/
│       └── filter_params.yaml
│
├── mapping/                         ← Your mapping layer
│   ├── launch/
│   │   └── mapping.launch.py
│   ├── src/
│   │   └── occupancy_mapper.py      ← Subscribes: /turtlebot/scan
│   │                                  Publishes: /occupancy_grid
│   └── config/
│       └── map_params.yaml
│
├── planning/                        ← Your planning layer
│   ├── launch/
│   │   └── planner.launch.py
│   ├── src/
│   │   ├── astar_planner.py         ← Subscribes: /occupancy_grid, /goal_pose
│   │   │                               Publishes: /path
│   │   └── efe_planner.py           ← Your novel approach
│   └── config/
│       └── planner_params.yaml
│
├── control/                         ← Your control layer
│   ├── launch/
│   │   └── controller.launch.py
│   ├── src/
│   │   ├── path_follower.py         ← Subscribes: /path, /turtlebot/odom
│   │   │                               Publishes: /turtlebot/cmd_vel
│   │   └── safety_shield.py         ← Subscribes: /turtlebot/scan
│   │                                  Modifies: /turtlebot/cmd_vel
│   └── config/
│       └── control_params.yaml
│
└── experiments/                     ← Your experiments
    ├── launch/
    │   └── run_experiment.launch.py ← Orchestrates everything
    │
    ├── scenarios/
    │   ├── scenario_1.yaml
    │   ├── scenario_2.yaml
    │   └── scenario_3.yaml
    │
    └── results/
        ├── experiment_1/
        └── experiment_2/
```

---

## How Everything Connects: The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ experiments/run_experiment.launch.py (Orchestrator)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Launches:                                                      │
│  1. sim/gazebo.launch.py                                        │
│     └─ Gazebo + TurtleBot3 + Bridge                            │
│                                                                 │
│  2. visualization/visualization.launch.py                       │
│     └─ RViz with full_pipeline.rviz config                     │
│                                                                 │
│  3. perception/perception.launch.py                             │
│  4. estimation/estimator.launch.py                              │
│  5. mapping/mapping.launch.py                                   │
│  6. planning/planner.launch.py                                  │
│  7. control/controller.launch.py                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ GAZEBO SIMULATION                                               │
├─────────────────────────────────────────────────────────────────┤
│ TurtleBot3 in warehouse.world                                   │
│ ├─ base_link (chassis)                                          │
│ ├─ laser_frame (sensor)                                         │
│ └─ wheels (actuators)                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
      ┌───────────────────────┴───────────────────────┐
      ↓                                               ↓
SENSOR DATA                               CONTROL INPUT
├─ /turtlebot/scan                       ├─ /turtlebot/cmd_vel
├─ /turtlebot/odom                       └─ (from control/control.py)
├─ /turtlebot/joint_states
└─ /tf (transforms)
      ↓
PERCEPTION LAYER
└─ perception/vision_pose_node.py
   └─ Publishes: /vision/obstacles, /vision/pose
                      ↓
              ESTIMATION LAYER
              └─ estimation/state_estimator.py
                 Subscribes: /turtlebot/odom, /vision/*
                 Publishes: /state/belief
                      ↓
              MAPPING LAYER
              └─ mapping/occupancy_mapper.py
                 Subscribes: /turtlebot/scan
                 Publishes: /occupancy_grid, /costmap
                      ↓
              PLANNING LAYER
              ├─ planning/astar_planner.py (baseline)
              │  Subscribes: /occupancy_grid, /goal_pose (from RViz)
              │  Publishes: /path
              │
              └─ planning/efe_planner.py (your thesis)
                 Subscribes: /state/belief, /occupancy_grid, /goal_pose
                 Publishes: /path, /planning_uncertainty
                      ↓
              CONTROL LAYER
              ├─ control/path_follower.py
              │  Subscribes: /path, /turtlebot/odom
              │  Publishes: /turtlebot/cmd_vel
              │
              └─ control/safety_shield.py
                 Subscribes: /turtlebot/scan, /turtlebot/cmd_vel
                 Modifies: /turtlebot/cmd_vel (if obstacle too close)
                      ↓
              [Back to GAZEBO for next iteration]

VISUALIZATION (RViz)
├─ Subscribes to ALL topics
├─ Loads full_pipeline.rviz config showing:
│  ├─ Robot model (from /tf)
│  ├─ Laser scan (/turtlebot/scan)
│  ├─ Occupancy grid (/occupancy_grid)
│  ├─ Costmap (/costmap)
│  ├─ Planned path (/path)
│  ├─ Belief uncertainty (custom markers)
│  └─ Trajectory history (/turtlebot/odom)
│
└─ Interactive tools:
   ├─ "Set 2D Goal" → /goal_pose
   ├─ "Set Initial Pose" → /initialpose
   └─ Drag robot with mouse (via control layer)
```

---

## Key Organizational Principles

### 1. **Namespace Isolation**
```python
# All robot-related topics use namespace
/turtlebot/cmd_vel    ← Input command
/turtlebot/odom       ← Output sensor
/turtlebot/scan       ← Output sensor

# Global planning topics (no namespace)
/occupancy_grid       ← Shared perception
/goal_pose            ← Global goal
/path                 ← Global path
```

### 2. **Launch Hierarchy**
```
run_experiment.launch.py (main orchestrator)
├─ Includes: gazebo.launch.py
├─ Includes: visualization.launch.py
├─ Includes: perception.launch.py
│   └─ May include sub-launchers
├─ Includes: planning.launch.py
└─ Includes: control.launch.py
```

### 3. **Configuration Over Code**
```
Launch file (*.launch.py)     ← Orchestration
    ↓
Parameter YAML files          ← Node-specific tuning
    ↓
Source code (*.py)            ← Logic (rarely changed)
```

### 4. **RViz Configs Match Pipeline Stages**
```
perception.rviz       ← Shows perception layer outputs
mapping.rviz          ← Shows occupancy grid + costmap
planning.rviz         ← Shows planner expansions + path
control.rviz          ← Shows velocity commands + safety
full_pipeline.rviz    ← Everything integrated (default)
```

### 5. **Sensor Data Flow**
```
Gazebo Sensors → ROS-Gazebo Bridge → ROS 2 Topics
                (via ros_gz_bridge)
                
Bridge translates:
/model/{robot}/scan (Gazebo) → /{robot}/scan (ROS)
/model/{robot}/odom (Gazebo) → /{robot}/odom (ROS)
/model/{robot}/cmd_vel (ROS) → /{robot}/cmd_vel (Gazebo)
```

---

## Summary Table: Component Relationships

| Component | Package | Input Topics | Output Topics | Config | Launch |
|-----------|---------|--------------|----------------|--------|--------|
| **Gazebo** | sim | - | /tf, /clock | worlds/*.world | gazebo.launch.py |
| **Robot Bridge** | sim | /turtlebot/cmd_vel | /turtlebot/odom, /scan | - | turtlebot3.launch.py |
| **Perception** | perception | /turtlebot/scan | /vision/obstacles | perception_params.yaml | perception.launch.py |
| **Estimation** | estimation | /turtlebot/odom, /vision/* | /state/belief | filter_params.yaml | estimator.launch.py |
| **Mapping** | mapping | /turtlebot/scan | /occupancy_grid | map_params.yaml | mapping.launch.py |
| **Planning (A\*)** | planning | /occupancy_grid, /goal_pose | /path | planner_params.yaml | planner.launch.py |
| **Planning (EFE)** | planning | /state/belief, /occupancy_grid, /goal_pose | /path, /uncertainty | efe_params.yaml | planner.launch.py |
| **Control** | control | /path, /turtlebot/odom, /turtlebot/scan | /turtlebot/cmd_vel | control_params.yaml | controller.launch.py |
| **RViz** | visualization | All topics | Interactive markers | full_pipeline.rviz | visualization.launch.py |

---

*Last Updated: January 29, 2026*

Provides reusable utilities for all other packages.

### 1.1 Visualization & Interaction

**`core_rviz_tools`** (RViz Configurations)
- 15+ pre-configured `.rviz` files for different visualization scenarios
- Generic RViz launcher with customizable parameters
- Supports robot model, scan, map, costmap, path visualization
- See [RVIZ_CONFIGURATION_GUIDE.md](RVIZ_CONFIGURATION_GUIDE.md) for details

**`core_plot_tools`** (Matplotlib Visualization)
- Real-time plotting of trajectories and plan visualizations
- Used in lectures for visualization of planning results

**`core_debug_tools`** (Debugging Utilities)
- Message logging and inspection
- Performance profiling helpers

### 1.2 Geometric & Transform Tools

**`core_geometry_tools`** (Geometry Utilities)
- 2D/3D geometric operations
- Point, line, polygon operations
- Collision checking utilities

**`core_tf_tools`** (Transform Frame Tools)
- Transform frame management
- Static/dynamic frame creation
- TF tree utilities

**`core_occupancy_grid_tools`** (Occupancy Grid Utilities)
- Grid creation and manipulation
- Grid visualization
- Grid-based algorithms

### 1.3 ROS Programming Utilities

**`core_launch_tools`** (Launch File Helpers)
- Reusable launch functions
- Package discovery utilities
- Parameter loading helpers

**`core_package_template`** (Package Template)
- Template for creating new packages
- Pre-configured structure (src/, launch/, config/)

**`core_my_package`** (Example Package)
- Demonstrates ROS 2 Python package structure
- Example node implementations

### 1.4 System Support

**`core_occupancy_grid_tools`** (Grid Tools)
- Grid representation and algorithms
- Grid-to-map conversions

---

## Layer 2: Simulation (core_robot_simulation_humble)

Provides Gazebo-based simulation infrastructure.

### 2.1 Gazebo Environment

**`core_gazebo_worlds`**
- Pre-configured Gazebo worlds
- Robot models (RoboCyl, TurtleBot3)
- Sensor configurations (laser scanner, camera)
- Static environmental objects

**`core_robot_simulate`** (Meta-package)
- Gazebo launcher with configurable parameters
- Robot spawning (RoboCyl variants: RoboCyl, RobobOX, RoboCylinder)
- Sensor simulators (laser, odometry, pose ground truth)
- RViz integration
- Teleop support (keyboard, joystick)

### 2.2 ROS-Gazebo Bridge

**`core_gazebo_teleport`**
- Teleport entities in Gazebo
- Used for repositioning robots and goals

**`core_ros_gz_service_bridge`**
- ROS 2 ↔ Gazebo service bridges
- Custom service interfaces for Gazebo

### 2.3 Key Launch Patterns

**Simulation Structure:**

```
Gazebo World
    ├── RoboCyl Robot
    │   ├── Laser Scanner (sensor)
    │   ├── Odometry Publisher
    │   └── Cmd_vel Subscriber
    │
    └── Environment Objects
        ├── Static walls
        ├── Obstacles
        └── Markers
```

**Example Launch Arguments:**

```bash
ros2 launch core_robot_simulate robocyl_gazebo_world.launch.py \
  gazebo_world:=gzworld \
  gazebo_world_model:=core_office \
  robot_namespace:=robocyl \
  use_sim_time:=true
```

---

## Layer 3: Control (core_robot_control_humble)

Implements low-level to mid-level robot control algorithms.

### 3.1 Control Algorithms

**`core_unicycle_control`**
- Unicycle kinematic model
- Control node: converts velocity commands to motor controls
- Teleop nodes: keyboard and joystick teleoperations

**`core_path_follow_ctrl`**
- Path-following controllers
- Pure pursuit algorithm
- Local trajectory tracking

**`core_scan_gap_follower`**
- Reactive gap-following control
- Used for obstacle avoidance
- Follows largest gap in laser scan

**`core_goal_nav_ctrl`**
- Goal-reaching control
- Direct navigation toward goal

**`core_apf_nav_ctrl`** (Artificial Potential Fields)
- APF-based collision avoidance
- Goal attraction + obstacle repulsion

**`core_cvx_opt_ctrl`** (Convex Optimization)
- Optimization-based control
- Quadratic programming solvers

**`core_proj_nav_ctrl`** (Projection-based)
- Projection methods for control
- Constrained navigation

### 3.2 Twist & Pose Management

**`core_twist_tools`** (Velocity Command Tools)
- Teleop (keyboard, joystick) → twist commands
- Twist multiplexing (arbitrate between multiple sources)
- Twist integration to odometry

**`core_pose_tools`** (Pose Management)
- Pose teleop (keyboard control of robot pose)
- Pose multiplexing (arbitrate between controllers)

### 3.3 Control Data Flow

```
User Input (Teleop or Planner)
    ↓
Control Node (e.g., path_follow_ctrl)
    ↓
Twist Command (geometry_msgs/Twist)
    ↓
Robot Model / Gazebo
    ↓
Odometry Output
```

---

## Layer 4: Planning (core_robot_planning_humble)

Implements motion planning and mapping algorithms.

### 4.1 Occupancy Grid & Costmap

**`core_occupancy_grid_costmap`**
- Occupancy grid map creation from sensor data
- Costmap generation with obstacle inflation
- Distance transform computation
- Grid-based cost visualization

**`core_map_tools`** (Map Management)
- Map server integration
- YAML map file management
- Map frame transformations

### 4.2 Path Planning

**`core_search_path_planner`** (Grid-based Search)
- A* algorithm implementation
- Dijkstra's algorithm
- Breadth-first search variants
- Grid-based path planning
- **Topics:**
  - Input: `/occupancy_grid`, `/start_pose`, `/goal_pose`
  - Output: `/path` (nav_msgs/Path)

**`core_sampling_path_planner`** (Sampling-based)
- RRT (Rapidly-exploring Random Trees)
- RRT* variants
- PRM (Probabilistic Roadmap)
- Configuration space sampling

### 4.3 Path & Goal Tools

**`core_path_tools`** (Path Utilities)
- Path smoothing
- Path simplification
- Path distance metrics

**`core_goal_tools`** (Goal Management)
- Goal markers and visualization
- Goal tolerance checking

**`core_nav2_tools`** (Nav2 Integration)
- Integration with ROS 2 Navigation2 stack
- Behavior tree interfaces

### 4.4 Planning Data Flow

```
Sensor Data (LaserScan)
    ↓
Occupancy Grid Creation
    ↓
Costmap Generation
    ↓
Path Planner (A*, RRT, etc.)
    ↓
Path Output (nav_msgs/Path)
    ↓
Control (Path Following)
```

---

## Layer 5: Robots (Robot-Specific Wrappers)

Provide simplified interfaces for specific robots.

### 5.1 TU Eindhoven Robots

**`core_tue4tm00_robot_humble`**
- Wraps both core_robot_* and lectures
- Provides unified launch interface
- TU Eindhoven custom configurations

**`core_tue4tm00_robot_control`**
- Custom control implementations
- Lab-specific controllers

**`core_tue4tm00_robot_simulate`**
- Lab robot simulation wrappers
- Integration with lab infrastructure

### 5.2 TurtleBot3 Support

**`core_tue4tm00_turtlebot3_humble`**
- TurtleBot3 adaptation layer
- Uses standard TurtleBot3 URDF
- Custom launch configurations

**`core_turtlebot3_control_humble`**
- Generic TurtleBot3 control packages

**`core_turtlebot3_simulation_humble`**
- Generic TurtleBot3 simulation

---

## Layer 6: Lectures & Assignments (Educational Content)

### 6.1 Lecture Modules (9 total)

**`lecture_ros_intro`** (ROS 2 Introduction)
- Basic ROS 2 concepts
- Node communication
- Publisher/Subscriber pattern

**`lecture_ros_programming`** (ROS 2 Programming)
- Advanced ROS 2 features
- Service/Action clients
- Parameter servers

**`lecture_geometric_robot_control`** (Control Theory)
- Kinematic models
- Dynamic models
- Control law derivations

**`lecture_odometry`** (Localization Basics)
- Dead reckoning
- Sensor fusion
- Pose estimation

**`lecture_path_following`** (Path Following Control)
- Pure pursuit algorithm
- Trajectory tracking
- LQR-based controllers
- Interactive demonstrations

**`lecture_feedback_robot_control`** (Feedback Control)
- PID control
- State feedback
- Observer design

**`lecture_optimal_control`** (Optimal Control)
- Dynamic programming
- MPC (Model Predictive Control)
- LQR optimal control

**`lecture_sampling_path_planner`** (Sampling-based Planning)
- RRT algorithms
- Probabilistic methods
- Interactive path planning

**`lecture_search_path_planner`** (Search-based Planning)
- Grid-based search
- A*, Dijkstra
- Heuristic planning

### 6.2 Assignments (3 total, Progressive)

**`core_tue4tm00_assignment1`** (Basics)
- Simple control task
- Velocity command generation
- Basic path following

**`core_tue4tm00_assignment2`** (Intermediate)
- Path planning from occupancy grid
- Obstacle avoidance
- Planning + control integration

**`core_tue4tm00_project`** (Final)
- Complete autonomous navigation
- Multi-robot scenarios
- Advanced planning/control features

### 6.3 Lecture Structure Pattern

Each lecture package contains:

```
lecture_*/
├── core_lec_*/                    # Core implementation package
│   ├── launch/                    # Demo launch files
│   ├── config/                    # RViz configs, parameter files
│   ├── src/                       # Implementation nodes
│   └── scripts/                   # Utility scripts
└── README.md                      # Lecture documentation
```

**Example: lecture_path_following**

```
lecture_path_following/
├── core_lec_path_follow_ctrl/
│   ├── launch/
│   │   ├── demo_plot_path_goal_circular_corridor.launch.py
│   │   └── demo_plot_path_goal_support_corridor.launch.py
│   ├── config/
│   │   ├── robot_scan_path.rviz        # RViz config
│   │   ├── controller_params.yaml
│   │   └── corridor_*.yaml             # Environment configs
│   └── src/
│       ├── path_follower_node.py
│       └── corridor_generator.py
└── README.md
```

---

## Critical Design Patterns

### Pattern 1: Modular Launch Hierarchy

Launches are composed hierarchically using `GroupAction` and `IncludeLaunchDescription`:

```python
# Parent launch
rviz_launch = GroupAction(
    actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(...)
        )
    ],
    launch_configurations={
        'config': LaunchConfiguration('rviz_config'),
        'use_sim_time': LaunchConfiguration('use_sim_time'),
    },
)
ld.add_action(rviz_launch)
```

**Benefits:**
- Reusable launch files
- Parameterized behavior
- Easy to compose complex systems

### Pattern 2: Namespace Isolation

Each subsystem uses ROS namespaces for isolation:

```
/gazebo/              # Gazebo node
/robocyl/             # Robot namespace
  /cmd_vel            # Input velocity
  /odom               # Odometry
  /scan               # Laser scan
  /tf                 # Transforms
/planner/             # Planning node
/controller/          # Control node
/rviz/                # Visualization
```

**Benefits:**
- Multi-robot support
- Encapsulation
- Avoid topic name collisions

### Pattern 3: Configuration Over Code

Heavy use of YAML configuration files:

```
*.launch.py           # Launch logic
config/
├── *.rviz            # RViz configs
├── *.yaml            # Node parameters
├── *.world           # Gazebo worlds
└── *.model           # Robot/object models
```

**Benefits:**
- Easy experimentation
- Reproducibility
- No code recompilation needed

### Pattern 4: Sensor-to-Plan Pipeline

Clear data flow from sensors to planning to control:

```
Sensors (Gazebo)
    ↓ [sensor_msgs]
Occupancy Grid Mapper
    ↓ [nav_msgs/OccupancyGrid]
Path Planner
    ↓ [nav_msgs/Path]
Path Follower
    ↓ [geometry_msgs/Twist]
Robot Controller
    ↓ [robot action]
Actuators (Gazebo)
```

---

## Key ROS Communication Patterns

### Topics Used Throughout

| Topic | Message Type | Direction | Purpose |
|---|---|---|---|
| `/scan` | sensor_msgs/LaserScan | ← Sensor | Lidar data |
| `/odom` | nav_msgs/Odometry | ← Odometry | Wheel odometry |
| `/map` | nav_msgs/OccupancyGrid | → Map | Static map |
| `/occupancy_grid` | nav_msgs/OccupancyGrid | ← Mapper | Dynamic occupancy |
| `/costmap` | nav_msgs/OccupancyGrid | ← Costmap | Inflated obstacles |
| `/plan` | nav_msgs/Path | ← Planner | Planned path |
| `/cmd_vel` | geometry_msgs/Twist | → Control | Velocity command |
| `/goal_pose` | geometry_msgs/PoseStamped | ← Input | Target goal |
| `/tf` | tf2_msgs/* | ← TF | Transform frames |
| `/initialpose` | geometry_msgs/PoseWithCovarianceStamped | ← Input | Initial pose |

### Services Used

| Service | Purpose |
|---|---|
| `/gazebo/spawn_entity` | Spawn entities in Gazebo |
| `/gazebo/delete_entity` | Remove entities |
| `/gazebo/set_entity_state` | Teleport entities |
| `map_server/load_map` | Load map from YAML |

---

## Typical Workflow: Running a Demo

### Example: Autonomous Navigation with A* Planning

```bash
# Terminal 1: Launch Gazebo simulation
ros2 launch core_robot_simulate robocyl_gazebo_world.launch.py \
  gazebo_world_model:=core_office

# Terminal 2: Launch planning + RViz
ros2 launch core_search_path_planner demo_search_path_planner.launch.py

# Terminal 3: Launch control
ros2 launch core_path_follow_ctrl demo_path_follower.launch.py

# In RViz: Use "Set Goal Pose" tool to specify goal
# → Planner finds path
# → Controller follows path
# → Robot reaches goal
```

### Data Flow During Demo

```
RViz Tool (User Input)
    ↓ /goal_pose
Path Planner
    ↓ /plan
Path Follower
    ↓ /cmd_vel
Gazebo Robot
    ↓ /odom, /scan
RViz Visualization
```

---

## Dependencies Summary

### System Level

- **OS:** Ubuntu 22.04 LTS
- **ROS:** ROS 2 Humble
- **Simulator:** Gazebo Fortress
- **Build:** ament_cmake/ament_python
- **Python:** Python 3.10+

### ROS 2 Packages (Key)

```
rclpy, rclcpp              # ROS client libraries
sensor_msgs                # Sensor data
geometry_msgs              # Geometry (Twist, Pose)
nav_msgs                   # Navigation (Path, OccupancyGrid)
tf2_ros                    # Transform frames
visualization_msgs        # Markers
ros_gz                     # ROS-Gazebo bridge
rviz2                      # Visualization
launch_ros                 # Launch system
```

### External Python

```
numpy                      # Numerical computing
scipy                      # Scientific computing
networkx                   # Graph algorithms
transforms3d               # 3D transformations
trimesh                    # Mesh processing
matplotlib                 # Plotting
```

---

## Package Dependency Graph

### High-Level Dependencies

```
Lectures & Assignments
    ↓
[Control Humble, Planning Humble, Simulation Humble, Robot Humble]
    ↓
ROS 2 Humble + Gazebo Fortress + Utility Tools
```

### Control Humble Dependencies

```
core_path_follow_ctrl
    ← core_pose_tools
    ← core_twist_tools
    ← ROS core
```

### Planning Humble Dependencies

```
core_search_path_planner
    ← core_occupancy_grid_costmap
    ← numpy, networkx
    ← ROS core
```

### Simulation Humble Dependencies

```
core_robot_simulate
    ← core_gazebo_worlds
    ← core_gazebo_teleport
    ← ros_gz, Gazebo Fortress
```

---

## Development Workflow

### Adding a New Feature

**Step 1: Choose location**
- **Control?** → `core_robot_control_humble/`
- **Planning?** → `core_robot_planning_humble/`
- **Utility?** → `core_robot_humble/`
- **Educational?** → `core_tue4tm00_lectures_humble/`

**Step 2: Create package**

```bash
cd core_tue4tm00_humble
ros2 pkg create my_package --build-type ament_python
```

**Step 3: Follow structure**

```
my_package/
├── launch/
│   └── demo_my_feature.launch.py
├── config/
│   ├── params.yaml
│   └── my_feature.rviz
├── src/
│   └── my_node.py
├── package.xml
└── README.md
```

**Step 4: Build and test**

```bash
cd ~/tue4tm00_ws
colcon build --packages-select my_package
source install/local_setup.bash
ros2 launch my_package demo_my_feature.launch.py
```

### Testing a Module

```bash
# Run demo
ros2 launch <package> demo_*.launch.py

# Monitor topics
ros2 topic list
ros2 topic echo /topic_name

# Inspect RViz
ros2 topic info /topic_name

# Check node graph
rqt_graph
```

---

## Performance Characteristics

### Typical Resource Usage

| Component | CPU | Memory | Notes |
|---|---|---|---|
| Gazebo + RoboCyl | 20-30% | 400-500 MB | Single robot simulation |
| Path Planner (A*) | 5-10% | 100-200 MB | Grid 100×100 |
| RViz | 2-5% | 150-300 MB | With visualization |
| Path Follower | <1% | <50 MB | Control loop at 10-20 Hz |

### Scalability

- **Single robot:** Fully supported
- **Multiple robots:** Namespacing supports 2-3 robots
- **Large grids:** Grid planner works up to ~500×500

---

## Troubleshooting Common Issues

### Issue: Gazebo doesn't launch

```bash
# Solution: Check Gazebo installation
gazebo --version

# Install if missing
sudo apt install ros-humble-ros-gz

# Check OpenGL
glxinfo | grep "OpenGL version"
```

### Issue: RViz can't display robot model

```bash
# Check if robot_description is published
ros2 topic list | grep robot_description
ros2 topic echo /robot_description | head -20

# Ensure RobotModel display uses correct topic
```

### Issue: Path planner finds no solution

```bash
# Check occupancy grid
ros2 topic echo /occupancy_grid

# Check goal is not in obstacle
ros2 topic echo /goal_pose

# Check planner is receiving inputs
ros2 node info /planner_node
```

### Issue: Colcon build fails

```bash
# Check dependencies
rosdep check --from-paths src

# Install missing dependencies
rosdep install --from-paths src -y --ignore-src

# Clean and rebuild
rm -rf build install
colcon build --symlink-install
```

---

## Best Practices for Using This Codebase

### 1. **Always Use Namespaces**

```python
# Good
namespace = LaunchConfiguration('namespace')
node = Node(namespace=namespace, ...)

# Avoid
node = Node(...)  # No namespace → potential conflicts
```

### 2. **Document Topic Names**

```python
# Good
"""
Node: path_follower
Subscribes:
  - /plan (nav_msgs/Path): Planned path to follow
  - /odom (nav_msgs/Odometry): Current odometry
Publishes:
  - /cmd_vel (geometry_msgs/Twist): Velocity command
"""

# Avoid: Undocumented topics
```

### 3. **Use Configuration Files**

```python
# Good - parameterizable
params = {'v_max': 0.5, 'w_max': 1.0}
node = Node(parameters=[params])

# Avoid - hardcoded
v_max = 0.5  # In code, hard to change
```

### 4. **Reuse Launch Files**

```python
# Good - reusable
rviz_launch = include_rviz_launch(config=rviz_config)

# Avoid - duplicating launch code
```

### 5. **Set Reasonable Defaults**

```python
# Good
rate = DeclareLaunchArgument('rate', default_value='10.0')

# Avoid
# Missing default → launch fails if arg not provided
```

---

## Future Extensions (for Thesis Integration)

The codebase is designed to be extensible. For your thesis repository (`tue_ai_es_thesis_ws`):

1. **Perception layer:** Add vision nodes reusing robot models
2. **Estimation layer:** Add Bayesian filters on top of odometry
3. **Planning:** Create EFE planner alongside A*
4. **Visualization:** Create custom markers for beliefs, uncertainties
5. **Experiments:** Use assignment structure as template

See [RVIZ_CONFIGURATION_GUIDE.md](RVIZ_CONFIGURATION_GUIDE.md) for visualization integration.

---

## Navigation Guide

### For Learning ROS 2
→ Start with `lecture_ros_intro` and `lecture_ros_programming`

### For Learning Control
→ `lecture_geometric_robot_control` → `lecture_path_following` → `lecture_feedback_robot_control`

### For Learning Planning
→ `lecture_sampling_path_planner` → `lecture_search_path_planner`

### For Implementing Features
→ Use `core_robot_control_humble/` for control  
→ Use `core_robot_planning_humble/` for planning  
→ Use `core_robot_humble/` for utilities

### For Running Experiments
→ Use lecture demos as templates  
→ Adapt assignment launchers for custom scenarios

---

## Summary Table: Package Organization

| Directory | Purpose | Packages | Key Technologies |
|---|---|---|---|
| **core_robot_control_humble/** | Robot control | 10 | Twist, control laws, teleop |
| **core_robot_planning_humble/** | Motion planning | 7 | A*, RRT, occupancy grids |
| **core_robot_simulation_humble/** | Gazebo simulation | 4 | Gazebo Fortress, ROS-Gz bridge |
| **core_robot_humble/** | Utilities | 11 | TF, debugging, visualization |
| **core_tue4tm00_robot_humble/** | TUE wrappers | 2 | Integration, custom configs |
| **core_tue4tm00_turtlebot3_humble/** | TurtleBot3 wrappers | 2 | TurtleBot3 adaptation |
| **core_turtlebot3_*_humble/** | Generic TurtleBot | 2 | Standard TurtleBot packages |
| **core_tue4tm00_lectures_humble/** | Educational | 12 | Lectures + assignments |

---

## Conclusion

The `core_tue4tm00_humble` repository provides a well-structured, educational foundation for robot motion planning and control. Its modular design, clear separation of concerns, and comprehensive documentation make it an excellent template for building thesis-specific research extensions.

For your thesis work in `tue_ai_es_thesis_ws`, this repository serves as:
1. **Reference architecture** for package organization
2. **Base implementations** for simulation and control
3. **Educational foundation** for understanding ROS 2
4. **Development pattern** for extending functionality

---

*Last Updated: January 29, 2026*  
*Related Documents: [RVIZ_CONFIGURATION_GUIDE.md](RVIZ_CONFIGURATION_GUIDE.md)*
