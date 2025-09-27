
<<<<<<< HEAD
# gazebo-simulation-of-turtlebot
=======
Project Structure
ros2_ws/
└── src/
    └── marl/
        ├── worlds/
        │   └── maze.world
        ├── rl_training/
        │   └── scripts/
        │       └── train_turtlebot.py
        ├── models/
        ├── launch/
        ├── urdf/
        └── test/

<<<<<<< HEAD
        1. Prerequisites
=======
⚙️ Installation & Setup
1. Prerequisites

Ubuntu 22.04 LTS (recommended)

ROS2 Humble or newer

Gazebo (Fortress or Garden)

Python 3.10+ with ML libraries (PyTorch / TensorFlow depending on your training code)

2. Clone & Build
# Create workspace
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# Clone this repo
git clone https://github.com/sakina998901/gazebo-simulation-of-turtlebot

# Build
cd ~/ros2_ws
colcon build

# Source
source install/setup.bash

3. Run Simulation
# Launch Gazebo with maze world and TurtleBot3
ros2 launch marl gazebo_maze.launch.py

4. Train RL Agent
python3 src/marl/rl_training/scripts/train_turtlebot.py
nano README.md
