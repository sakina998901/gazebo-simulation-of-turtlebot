import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo with maze world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', os.path.expanduser('~/ros2_ws/src/marl/worlds/maze.world')],
            output='screen'
        ),

        # Spawn TurtleBot3 in Gazebo maze
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'turtlebot3',
                '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf',
                '-x', '0', '-y', '0', '-z', '0.1'
            ],
            output='screen'
        )
    ])
