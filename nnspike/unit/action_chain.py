import time
from typing import Optional, Tuple

import numpy as np

from nnspike.constants import OFFSET_Y, ROI_CNN, Mode
from nnspike.unit.etrobot import ETRobot
from nnspike.utils import find_bottle_center, get_line_edges_at_y


class ActionChain(object):
    """
    A class to manage a sequence of actions for an ETRobot.

    This class allows you to define a chain of actions, each consisting of
    setting left and right motor speeds for a specified duration.
    """

    def __init__(self, et: ETRobot, course: str) -> None:
        self.et = et
        self.course = course
        self.start_time = 0.0
        self.current_time = 0.0

    def avoid_obstacle(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        backup/20250724/actions.pyと同じ障害物回避アクションチェーンを実行する。
        左旋回(40,70,0.8s)→右旋回(80,50,1.3s)→通常復帰
        """
        if not hasattr(self, '_obstacle_state'):
            self._obstacle_state = None
        action_chain = [
            (40, 70, 0.8),  # 左旋回
            (80, 50, 1.3),  # 右旋回
        ]
        state = self._obstacle_state
        if state is None:
            idx = 0
            left, right, duration = action_chain[0]
            self._obstacle_state = {'idx': 0, 'remain': duration}
            return None, (left, right), Mode.AVOID_OBSTACLE
        idx = state['idx']
        remain = state['remain']
        dt = 0.05  # 1フレーム分進める
        remain -= dt
        if remain > 0:
            self._obstacle_state = {'idx': idx, 'remain': remain}
            left, right, duration = action_chain[idx]
            return None, (left, right), Mode.AVOID_OBSTACLE
        # 次のアクションへ
        idx += 1
        if idx >= len(action_chain):
            self._obstacle_state = None
            return None, None, Mode.FOLLOW_LEFT_EDGE if self.course == "left" else Mode.FOLLOW_RIGHT_EDGE
        left, right, duration = action_chain[idx]
        self._obstacle_state = {'idx': idx, 'remain': duration}
        return None, (left, right), Mode.AVOID_OBSTACLE

    def heading_bottle1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        Perform a sequence of actions to head towards bottle 1.

        Args:
            et (ETRobot): The ETRobot instance to control.
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")

        if blue_pixel_count < 1000:
            left_x, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)

            if left_x is not None and right_x is not None:
                if right_x - left_x < 100:
                    return (left_x + right_x) / 2, None, Mode.HEAD_BOTTLE1
            else:
                return left_x, None, Mode.HEAD_BOTTLE1

        status = self.et.get_spike_status()
        # If the distance to the bottle is less than 1cm, pause
        if status.sensors.distance is not None and status.sensors.distance < 0.01:
            return None, None, Mode.PAUSE

        if center is not None:
            cx, _ = center
        else:
            cx = None

        return cx, None, Mode.HEAD_BOTTLE1

    def carry_bottle1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        self.start_time = time.time() if self.start_time is None else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time

        if elapsed_time < 0.8:
            left_speed, right_speed = 70, 70
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1
        elif elapsed_time > 0.8 and elapsed_time < 1.2:
            left_speed, right_speed = (70, 0) if self.course == "left" else (0, 70)
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1
        elif elapsed_time > 1.2 and elapsed_time < 2.0:
            left_speed, right_speed = 70, 70
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1
        elif elapsed_time > 2.0 and elapsed_time < 3.0:
            left_speed, right_speed = (70, 0) if self.course == "left" else (0, 70)
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1
        elif elapsed_time > 3.0 and elapsed_time < 4.0:
            _, _, _ = find_bottle_center(image=image, color="blue")
            left_speed, right_speed = 70, 70
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # Ensure a return value for all code paths
        return None, None, Mode.CARRY_BOTTLE1

    def heading_bottle2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        Perform a sequence of actions to head towards bottle 2.

        Args:
            et (ETRobot): The ETRobot instance to control.
        """
        raise NotImplementedError("This method should be implemented based on the specific behavior for heading towards bottle 2.")

    def carry_bottle2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        Perform a sequence of actions to carry bottle 2.

        Args:
            et (ETRobot): The ETRobot instance to control.
        """
        raise NotImplementedError("This method should be implemented based on the specific behavior for carrying bottle 2.")

    def heading_goal(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        Perform a sequence of actions to head towards the goal.

        Args:
            et (ETRobot): The ETRobot instance to control.
        """
        raise NotImplementedError("This method should be implemented based on the specific behavior for heading towards the goal.")

    def trun_left(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        左旋回アクションを実行する（backup/actions.pyのturn_left相当）。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.8:
            left_speed, right_speed = 0, 60
            return None, (left_speed, right_speed), Mode.LEFT_TURN if hasattr(Mode, 'LEFT_TURN') else Mode.AVOID_OBSTACLE
        return None, None, Mode.FOLLOW_LEFT_EDGE if hasattr(Mode, 'FOLLOW_LEFT_EDGE') else Mode.AVOID_OBSTACLE

    def trun_right(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        右旋回アクションを実行する（backup/actions.pyのturn_right相当）。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.8:
            left_speed, right_speed = 60, 0
            return None, (left_speed, right_speed), Mode.RIGHT_TURN if hasattr(Mode, 'RIGHT_TURN') else Mode.AVOID_OBSTACLE
        return None, None, Mode.FOLLOW_RIGHT_EDGE if hasattr(Mode, 'FOLLOW_RIGHT_EDGE') else Mode.AVOID_OBSTACLE
