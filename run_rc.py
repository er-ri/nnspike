#!/usr/bin/env python3
import sys
import time
import logging
from nnspike.unit import ETRobot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Platform-specific imports for keyboard input
try:
    import msvcrt  # Windows

    WINDOWS = True
except ImportError:
    import select
    import tty
    import termios

    WINDOWS = False

# User defined constants
BASE_SPEED = 55
SMOOTH_TURN_SPEED = 40  # For smoother turning


class KeyboardController:
    def __init__(self):
        self.running = True
        self.current_key = None

        if not WINDOWS:
            # Save terminal settings for Unix-like systems
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

    def get_key(self):
        """Get a single keypress"""
        if WINDOWS:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8").lower()
                return key
        else:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1).lower()
                return key
        return None

    def cleanup(self):
        """Restore terminal settings"""
        if not WINDOWS:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main():
    # Initialize robot and keyboard controller
    et = ETRobot()
    keyboard = KeyboardController()

    logger.info("Remote Control Car Initialized!")
    logger.info("Controls:")
    logger.info("  w - Move Forward")
    logger.info("  s - Move Backward")
    logger.info("  a - Smooth Turn Left")
    logger.info("  d - Smooth Turn Right")
    logger.info(
        "  o - Complete Obstacle Avoidance Maneuver (returns to original heading)"
    )
    logger.info("  b - Stop/Brake")
    logger.info("  q - Quit")
    logger.info("Press any key to start...")

    # Wait for initial keypress
    while True:
        key = keyboard.get_key()
        if key:
            break
        time.sleep(0.01)

    logger.info("Remote control active!")

    try:
        while keyboard.running:
            # Get keyboard input
            key = keyboard.get_key()

            if key:
                if key == "q":
                    logger.info("Quitting...")
                    break
                elif key == "w":
                    # Move forward
                    logger.info("Moving forward")
                    et.set_motor_forward_speed(
                        left_speed=BASE_SPEED, right_speed=BASE_SPEED
                    )
                elif key == "s":
                    # Move backward
                    logger.info("Moving backward")
                    et.set_motor_backward_speed(
                        left_speed=BASE_SPEED, right_speed=BASE_SPEED
                    )
                elif key == "a":
                    # Smooth turn left
                    logger.info("Turning left")
                    et.set_motor_forward_speed(
                        left_speed=SMOOTH_TURN_SPEED, right_speed=BASE_SPEED
                    )
                elif key == "d":  # Smooth turn right
                    logger.info("Turning right")
                    et.set_motor_forward_speed(
                        left_speed=BASE_SPEED, right_speed=SMOOTH_TURN_SPEED
                    )
                elif key == "b":
                    # brake
                    logger.info("Stopping")
                    et.brake()
                else:  # Unknown key - brake for safety
                    et.brake()

            # Small delay to prevent excessive CPU usage while maintaining responsiveness
            time.sleep(0.02)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping robot...")

    finally:
        logger.info("Cleaning up resources...")  # Stop the robot
        et.stop()  # Clean up keyboard controller
        keyboard.cleanup()

        logger.info("Cleanup completed.")


if __name__ == "__main__":
    main()
