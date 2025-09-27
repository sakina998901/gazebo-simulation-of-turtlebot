import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry  # Added for goal detection
import time
import random
import math  # Added for distance calculation

class GazeboMazeTurtleBotEnv(gym.Env):
    def __init__(self):
        super(GazeboMazeTurtleBotEnv, self).__init__()

        # Initialize ROS2 node
        rclpy.init(args=None)
        self.node = Node("turtlebot_env")

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Forward, Backward, Left, Right
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(360,), dtype=np.float32)  # Flattened to (360,)

        # ROS2 Publishers & Subscribers
        self.vel_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        # Add odometry subscription for goal detection
        self.odom_sub = self.node.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.lidar_ranges = np.zeros(360, dtype=np.float32)  # Flattened to (360,)
        self.done = False
        self.last_action = None  # To store the last action for avoiding loops
        
        # Goal parameters
        self.goal_x = -2.0
        self.goal_y = -2.0
        self.goal_threshold = 0.5  # Distance threshold to consider goal reached
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.goal_reached = False

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        print(f"[DEBUG] scan_callback: ranges length = {len(ranges)}")  # Print length of LIDAR ranges

        # Pad or trim the ranges to ensure it always has 360 elements
        if len(ranges) != 360:
            print(f"[WARN] LIDAR returned {len(ranges)} values instead of 360.")
            ranges = np.pad(ranges, (0, max(0, 360 - len(ranges))), mode='constant', constant_values=10.0)
            ranges = ranges[:360]  # Trim if needed
    
        self.lidar_ranges = np.clip(ranges, 0.0, 10.0)  # No need to reshape here, it's already (360,)

        # Check for collision (if any range is below 0.2 meters)
        if np.min(ranges) < 0.001:
            self.done = True
    
    # New callback for odometry to detect goal
    def odom_callback(self, msg):
        # Get robot position from odometry
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Calculate distance to goal
        distance_to_goal = math.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        
        # Check if robot reached the goal
        if distance_to_goal < self.goal_threshold:
            print("[INFO] Goal reached!")
            self.goal_reached = True
            self.done = True

    def step(self, action):
        twist = Twist()

        # Handle post-collision pause
        if self.done and not self.goal_reached:
            # Stop the robot completely for 5 seconds
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            rclpy.spin_once(self.node, timeout_sec=0.1)

            # Wait for 5 seconds to stabilize
            print("[INFO] Collision detected. Robot stopping for 5 seconds...")
            time.sleep(5)

            # After waiting, clear the done flag to resume movement
            self.done = False
            print("[INFO] Robot has stabilized and will resume movement.")

            # Choose a new random action to avoid repeated behavior
            action = random.choice([0, 1, 2, 3])

        # Execute action
        if action == 0:  # Forward
            twist.linear.x = 0.2
            twist.angular.z = 0.0
        elif action == 1:  # Backward
            twist.linear.x = -0.2
            twist.angular.z = 0.0
        elif action == 2:  # Turn left
            twist.linear.x = 0.0
            twist.angular.z = 0.5
        elif action == 3:  # Turn right
            twist.linear.x = 0.0
            twist.angular.z = -0.5

        # Publish movement command
        self.vel_pub.publish(twist)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        # Reward and termination
        reward = 1.0 if action == 0 else -0.5
        terminated = self.done
        truncated = False

        # Additional reward for reaching the goal
        if self.goal_reached:
            reward = 100.0
            print("[INFO] Goal reached! Rewarding agent.")
        elif self.done:  # Collision penalty
            reward = -10.0

        print(f"[DEBUG] lidar shape: {self.lidar_ranges.shape}")
        return self.lidar_ranges.astype(np.float32), reward, self.done, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.goal_reached = False  # Reset goal reached flag
        self.lidar_ranges = np.zeros(360, dtype=np.float32)  # (360,) shape
        rclpy.spin_once(self.node, timeout_sec=0.1)
        print(f"[DEBUG] reset() lidar_ranges shape before return: {self.lidar_ranges.shape}")
        return self.lidar_ranges.astype(np.float32), {}
    
    def render(self, mode="human"):
        pass

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()