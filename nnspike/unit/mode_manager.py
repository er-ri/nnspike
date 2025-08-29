from typing import Tuple

import numpy as np

from nnspike.constants import Mode
from nnspike.unit import ETRobot
from nnspike.utils import find_bottle_center


class ModeManager(object):
    """Manages the current mode of the robot unit based on image analysis and sensor data.

    This class is responsible for determining and transitioning between different behavior modes
    including edge following, obstacle avoidance, and various bottle carrying phases. Mode
    transitions are determined based on motor position thresholds, image processing results
    (color detection), and sensor readings.

    The class uses toggle flags to prevent repeated transitions and ensures each mode change
    occurs only once when conditions are met.
    """

    def __init__(self, course: str) -> None:
        """Initialize the ModeManager with a specified course direction.

        Args:
            course (str): The direction of the course, either "left" or "right"
        """
        self.course = course
        self.current_mode = Mode.FOLLOW_LEFT_EDGE if course == "left" else Mode.FOLLOW_RIGHT_EDGE
        self.previous_mode = self.current_mode
        self.mode_toggle_flag = {
            Mode.AVOID_OBSTACLE: False,
            Mode.CARRY_BOTTLE_PHASE1: False,
            Mode.CARRY_BOTTLE_PHASE6: False,
        }

    def get_current_mode(self) -> Mode:
        """
        Get the current mode of the action chain.
        """
        return self.current_mode

    def set_current_mode(self, mode: Mode) -> None:
        self.previous_mode = self.current_mode
        self.current_mode = mode

    def get_previous_mode(self) -> Mode:
        """
        Get the previous mode of the action chain.
        """
        return self.previous_mode

    def decide_next_mode(
        self,
        image: np.ndarray,
        et: ETRobot,
    ) -> Tuple[Mode, bool]:
        motors_relative_position = et.retrieve_motors_relative_position()

        if self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE2 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE3
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE3 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE4
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE5 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE6
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE6 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE7
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE7 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE8
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE8 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE9
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE9 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE10
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE10 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE11
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE12 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE13
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE13 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE14
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE15 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE16
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE16 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE17
        elif self.get_previous_mode() == Mode.CARRY_BOTTLE_PHASE18 and self.get_current_mode() == Mode.PHASE_COMPLETED:
            next_mode = Mode.CARRY_BOTTLE_PHASE19
        else:
            next_mode = self.get_current_mode()

        # Obstacle Avoidance
        if (
            motors_relative_position >= 5000
            and motors_relative_position < 8000
            and self.mode_toggle_flag[Mode.AVOID_OBSTACLE] is False
            and (self.get_current_mode() == Mode.FOLLOW_LEFT_EDGE or self.get_current_mode() == Mode.FOLLOW_RIGHT_EDGE)
        ):
            _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
            if yellow_pixel_count is not None and yellow_pixel_count > 2000:
                next_mode = Mode.AVOID_OBSTACLE
                self.mode_toggle_flag[Mode.AVOID_OBSTACLE] = True
        # Carry Bottle Phase 1
        elif (
            motors_relative_position >= 43000
            and motors_relative_position < 45000
            and self.mode_toggle_flag[Mode.CARRY_BOTTLE_PHASE1] is False
        ):
            _, _, red_pixel_count = find_bottle_center(image=image, color="red")
            if red_pixel_count is not None and red_pixel_count > 3000:
                next_mode = Mode.CARRY_BOTTLE_PHASE1
                self.mode_toggle_flag[Mode.CARRY_BOTTLE_PHASE1] = True
        # Carry Bottle Phase 5
        elif self.get_current_mode() == Mode.CARRY_BOTTLE_PHASE5:
            status = et.get_spike_status()
            color_reflected = status.sensors.color.reflected if status.sensors.color is not None else 0
            if color_reflected is not None and color_reflected < 500:
                next_mode = Mode.CARRY_BOTTLE_PHASE6
                self.mode_toggle_flag[Mode.CARRY_BOTTLE_PHASE6] = True
        # Carry Bottle Phase 14
        elif self.get_current_mode() == Mode.CARRY_BOTTLE_PHASE14:
            status = et.get_spike_status()
            color_reflected = status.sensors.color.reflected if status.sensors.color is not None else 0
            if color_reflected is not None and color_reflected < 900:
                next_mode = Mode.CARRY_BOTTLE_PHASE15
                self.mode_toggle_flag[Mode.CARRY_BOTTLE_PHASE15] = True
        # Carry Bottle Phase 17
        elif self.get_current_mode() == Mode.CARRY_BOTTLE_PHASE17:
            status = et.get_spike_status()
            color_reflected = status.sensors.color.reflected if status.sensors.color is not None else 0
            if color_reflected is not None and color_reflected < 500:
                next_mode = Mode.CARRY_BOTTLE_PHASE17
                self.mode_toggle_flag[Mode.CARRY_BOTTLE_PHASE17] = True
        # Carry Bottle Phase 19
        elif self.get_current_mode() == Mode.CARRY_BOTTLE_PHASE17:
            status = et.get_spike_status()
            color_reflected = status.sensors.color.reflected if status.sensors.color is not None else 0
            if color_reflected is not None and color_reflected < 800:
                next_mode = Mode.GOAL
                self.mode_toggle_flag[Mode.GOAL] = True

        return (next_mode, True) if next_mode != self.get_current_mode() else (self.get_current_mode(), False)
