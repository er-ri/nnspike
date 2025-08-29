import time
from typing import Optional, Tuple

import numpy as np

from nnspike.constants import Mode
from nnspike.unit.etrobot import ETRobot
from nnspike.utils import find_bottle_center


class ActionChain(object):
    """
    A class to manage a sequence of actions for an ETRobot.

    This class allows you to define a chain of actions, each consisting of
    setting left and right motor speeds for a specified duration.
    """

    def __init__(self, et: ETRobot, course: str) -> None:
        self.et = et
        self.course = course
        # Action with specific timing
        self.start_time = 0.0
        self.current_time = 0.0
        # Action with specific positioning
        self.start_position = 0
        self.current_position = 0
        self.mark_position = 0

    def reset(self) -> None:
        """
        Reset the action chain by setting the start position to 0.
        """
        self.start_time = 0.0
        self.current_time = 0.0

        self.start_position = 0
        self.current_position = 0
        self.mark_position = 0

    def follow_left_edge(
        self, image: np.ndarray, predicted_x: Optional[float]
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        return predicted_x, None, Mode.FOLLOW_LEFT_EDGE

    def follow_right_edge(
        self, image: np.ndarray, predicted_x: Optional[float]
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        return predicted_x, None, Mode.FOLLOW_RIGHT_EDGE

    def avoid_obstacle(self, init_flag: bool) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        Perform a sequence of actions to avoid an obstacle.

        Args:
            et (ETRobot): The ETRobot instance to control.
        """
        if init_flag is True:
            self.start_time = time.time()
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time

        if elapsed_time < 0.8:
            left_speed, right_speed = (70, 40) if self.course == "left" else (40, 70)
            return None, (left_speed, right_speed), Mode.AVOID_OBSTACLE

        elif elapsed_time > 0.8 and elapsed_time < 2.1:
            left_speed, right_speed = (50, 80) if self.course == "left" else (80, 50)
            return None, (left_speed, right_speed), Mode.AVOID_OBSTACLE

        self.reset()

        return None, None, Mode.FOLLOW_LEFT_EDGE if self.course == "left" else Mode.FOLLOW_RIGHT_EDGE

    def carry_bottle_phase1(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.start_position

        if moved_distance > 2200:  # Move forward 2200
            return predicted_x, None, Mode.CARRY_BOTTLE_PHASE1

        return predicted_x, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase2(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.start_position

        if moved_distance > 450:  # Turn left 450
            return None, (0, 40), Mode.CARRY_BOTTLE_PHASE2

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase3(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.start_position

        if moved_distance > 2900:  # Gate 2900
            return predicted_x, None, Mode.CARRY_BOTTLE_PHASE3

        return predicted_x, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase4(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.start_position

        if moved_distance > 450:  # Turn left 450
            return None, (0, 40), Mode.CARRY_BOTTLE_PHASE4

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase5(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        return predicted_x, None, Mode.CARRY_BOTTLE_PHASE5

    def carry_bottle_phase6(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.start_position

        if moved_distance > -450:  # Backward 450
            return None, (-40, -40), Mode.CARRY_BOTTLE_PHASE6

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase7(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.start_position

        if moved_distance > 950:
            return None, (40, 0), Mode.CARRY_BOTTLE_PHASE7

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase8(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        self.current_position = self.et.retrieve_motors_relative_position()

        moved_distance = self.current_position - self.mark_position

        if self.mark_position != 0 and moved_distance < 200:
            return predicted_x, None, Mode.CARRY_BOTTLE_PHASE8
        else:
            _, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count is not None and blue_pixel_count > 2000 and self.mark_position == 0:
                self.mark_position = self.et.retrieve_motors_relative_position()
                return predicted_x, None, Mode.CARRY_BOTTLE_PHASE8

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase9(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 900:
            return None, (40, 0), Mode.CARRY_BOTTLE_PHASE9

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase10(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 930:
            return predicted_x, None, Mode.CARRY_BOTTLE_PHASE10

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase11(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 380:
            return None, (40, 0), Mode.CARRY_BOTTLE_PHASE10

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase12(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 1790:
            return predicted_x, None, Mode.CARRY_BOTTLE_PHASE12

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase13(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 500:
            return None, (0, 40), Mode.CARRY_BOTTLE_PHASE13

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase14(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        return predicted_x, None, Mode.CARRY_BOTTLE_PHASE14

    def carry_bottle_phase15(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance < -500:
            return None, (-40, -40), Mode.CARRY_BOTTLE_PHASE15

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase16(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 370:
            return None, (40, 0), Mode.CARRY_BOTTLE_PHASE16

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase17(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        return predicted_x, None, Mode.CARRY_BOTTLE_PHASE17

    def carry_bottle_phase18(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        if init_flag is True:
            self.start_position = self.et.retrieve_motors_relative_position()
        self.current_position = self.et.retrieve_motors_relative_position()
        moved_distance = self.current_position - self.start_position

        if moved_distance > 390:
            return None, (0, 40), Mode.CARRY_BOTTLE_PHASE16

        return None, None, Mode.PHASE_COMPLETED

    def carry_bottle_phase19(
        self, image: np.ndarray, predicted_x: float, init_flag: bool
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        return predicted_x, None, Mode.CARRY_BOTTLE_PHASE19
