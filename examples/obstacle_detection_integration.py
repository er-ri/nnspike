"""
Example of how to integrate the obstacle detection function with the robot control system.

This script demonstrates how to use detect_two_obstacles_and_stop() in a robot control loop.
"""

import os
import sys
import time

# Add parent directory to path to import nnspike modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from nnspike.utils.control import detect_two_obstacles_and_stop, reset_obstacle_detection_state
from nnspike.unit.etrobot import ETRobot


def example_robot_control_with_obstacle_detection():
    """
    Example of robot control loop with obstacle detection integration.
    
    Note: This is a demonstration and requires actual robot hardware to run.
    """
    try:
        # Initialize robot (would require actual hardware)
        # robot = ETRobot(port="/dev/ttyACM0")  # Linux
        # robot = ETRobot(port="COM4")          # Windows
        
        print("Robot obstacle detection integration example")
        print("This would run with actual robot hardware")
        print()
        
        # Reset obstacle detection state at the beginning
        reset_obstacle_detection_state()
        print("✓ Obstacle detection state reset")
        
        # Example control loop (simulated)
        print("Example control loop structure:")
        print("""
# Main robot control loop
while robot_should_continue:
    # Check for obstacles and stop if two are detected
    if detect_two_obstacles_and_stop(robot):
        print("Robot stopped due to obstacle detection!")
        break
    
    # Continue with normal robot operations
    # robot.set_motor_forward_speed(left_speed, right_speed)
    # ... other robot control logic ...
    
    time.sleep(0.1)  # Control loop delay
        """)
        
        print("Integration points:")
        print("1. Call reset_obstacle_detection_state() when starting navigation")
        print("2. Call detect_two_obstacles_and_stop(robot) in your main control loop")
        print("3. The function returns True when robot is stopped due to obstacles")
        print("4. Customize thresholds: obstacle_threshold, detection_window, required_detections")
        
        print("\nCustomization example:")
        print("""
# Custom parameters for different scenarios
if detect_two_obstacles_and_stop(
    robot,
    obstacle_threshold=150,  # Detect obstacles within 15cm
    detection_window=3.0,    # Look for detections over 3 seconds
    required_detections=3    # Require 3 detections to stop
):
    print("Custom obstacle detection triggered!")
        """)
        
    except ImportError as e:
        print(f"Import error (expected without robot hardware): {e}")
        print("This example demonstrates integration patterns")


if __name__ == "__main__":
    example_robot_control_with_obstacle_detection()