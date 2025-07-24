import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import time

from nnspike.unit.etrobot import ETRobot


def main():
    print("Starting USB Connection to LEGO Spike Hub...")
    try:
        et = ETRobot(port="COM4")  # Windows example port, change as needed

<<<<<<< HEAD
        et.set_motor_forward_speed(left_speed=30, right_speed=30)  # Reset motors to ensure they are stopped
=======
        et.set_motor_speed(left_speed=30, right_speed=30)  # Reset motors to ensure they are stopped
>>>>>>> wip/teamwork-li

        # Print status every 0.5 seconds for 5 seconds
        end_time = time.time() + 5
        while time.time() < end_time:
            print(et.spike_status.motors["A"], et.spike_status.motors["B"])
            time.sleep(0.5)

        et.brake()  # Stop the motors

    except Exception as e:
        print(f"Error: {e}")
    finally:
        et.stop()


if __name__ == "__main__":
    main()
