import os
import sys

# Add parent directory to path to import nnspike modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import time
from nnspike.unit.etrobot import ETRobot

robot = ETRobot(port="/dev/ttyACM0")  # Use 'COM4' for Windows, '/dev/ttyACM0' for Linux

robot.move_arm(action=0, duration=2.0)  # Move arm up with speed 100
robot.move_arm(action=2)

print("Starting forward movement...")

start_time = time.time()
while time.time() - start_time < 5.0:
    # Set motors to move forward at moderate speed
    robot.set_motor_forward_speed(left_speed=50, right_speed=50)

    status = robot.get_spike_status()

    left_pos = status.motors["A"].relative_position
    right_pos = status.motors["B"].relative_position
    print(f"Time: {time.time() - start_time:.2f}s | Left motor: {left_pos} | Right motor: {right_pos}")
    time.sleep(0.1)  # Print every 200ms

robot.stop()  # Stop the robot
print("Stopping robot...")
