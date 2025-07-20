import time

from nnspike.unit.etrobot import ETRobot


def _perform_action_chain(action_chain: tuple[tuple[ETRobot, int, int, float], ...]) -> None:
    """
    Execute a sequence of motor control actions on an ETRobot with specified speeds and durations.

    Each action in the chain consists of setting left and right motor speeds for a specified duration.
    The function continuously monitors the robot's connection status and stops execution if
    the robot becomes disconnected.

    Args:
        action_chain (tuple): Tuple of tuples, where each inner tuple contains:
            - ETRobot: The robot instance to control
            - int: Left motor speed (-100-100)
            - int: Right motor speed (-100-100)
            - float: Duration in seconds to maintain these speeds

    Returns:
        None: Function returns early if robot disconnects during execution
    """
    for action in action_chain:
        et, left_speed, right_speed, duration = action

        start_time = time.time()
        while time.time() - start_time < duration:
            # Check if the robot is still connected
            if not et.is_running:
                print("ETRobot disconnected, stopping action chain.")
                return

            # Set motor speeds
            if left_speed >= 0 and right_speed >= 0:
                et.set_motor_forward_speed(left_speed, right_speed)
            elif left_speed <= 0 and right_speed <= 0:
                et.set_motor_backward_speed(abs(left_speed), abs(right_speed))
            else:
                raise ValueError("Both speeds must be either positive or negative.")

            time.sleep(0.05)  # Small delay to avoid overwhelming the robot


def avoid_obstacle(et: ETRobot) -> None:
    """
    Perform a sequence of actions to avoid an obstacle.

    Args:
        et (ETRobot): The ETRobot instance to control.
    """
    action_chain = (
        (et, 40, 70, 0.8),  # Turn left for 0.8 seconds
        (et, 80, 50, 1.3),  # Turn right for 1.3 seconds
    )

    _perform_action_chain(action_chain)
