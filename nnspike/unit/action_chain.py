import time
from typing import Optional, Tuple

import numpy as np

from nnspike.constants import OFFSET_Y, ROI_CNN, Mode
from nnspike.unit.etrobot import ETRobot
from nnspike.utils.control import find_bottle_center, get_line_edges_at_y, get_line_trace_edges_at_x320


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
        # 汎用的な状態管理用dict
        self._state = {}

    def avoid_obstacle(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        backup/20250724/actions.pyと同じ障害物回避アクションチェーンを実行する。
        左旋回(40,70,0.8s)→右旋回(80,50,1.3s)→通常復帰
        """
        # self.start_timeで経過時間を管理
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        elapsed_time = time.time() - self.start_time

        if elapsed_time < 0.8:
            left_speed, right_speed = 40, 70  # 左旋回
            return None, (left_speed, right_speed), Mode.AVOID_OBSTACLE
        elif elapsed_time < 2.1:  # 0.8 + 1.3
            left_speed, right_speed = 80, 50  # 右旋回
            return None, (left_speed, right_speed), Mode.AVOID_OBSTACLE
        else:
            self.start_time = 0.0  # チェーン終了でリセット
            return None, None, Mode.FOLLOW_LEFT_EDGE if self.course == "left" else Mode.FOLLOW_RIGHT_EDGE

    def heading_bottle1(self, image: np.ndarray) -> Tuple[Optional[float], Mode]:
        """
        Perform a sequence of actions to head towards bottle 1.

        Args:
            et (ETRobot): The ETRobot instance to control.
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        center, _, red_pixel_count = find_bottle_center(image=image, color="red")

        if red_pixel_count < 1000:
            left_x, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)

            if left_x is not None and right_x is not None:
                if right_x - left_x < 100:
                    return (left_x + right_x) / 2, Mode.HEAD_BOTTLE1
            else:
                return left_x, Mode.HEAD_BOTTLE1

        status = self.et.get_spike_status()
        # If the distance to the bottle is less than 1cm, pause
        if status.sensors.distance is not None and status.sensors.distance < 0.01:
            return None, Mode.PAUSE

        if center is not None:
            cx, _ = center
        else:
            cx = None

        return cx, Mode.HEAD_BOTTLE1

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
            return None, (left_speed, right_speed), Mode.TURN_LEFT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def trun_right(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        右旋回アクションを実行する（backup/actions.pyのturn_right相当）。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.8:
            left_speed, right_speed = 60, 0
            return None, (left_speed, right_speed), Mode.TURN_RIGHT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def small_turn_left(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        スモールターンレフト（短時間左旋回）アクション。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.3:
            left_speed, right_speed = 0, 50
            return None, (left_speed, right_speed), Mode.SMALL_TURN_LEFT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def small_turn_right(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        スモールターンライト（短時間右旋回）アクション。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.3:
            left_speed, right_speed = 50, 0
            return None, (left_speed, right_speed), Mode.SMALL_TURN_RIGHT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def turn_at_end(self, frame, offset_y_for_turn=400):
        """
        TURN_AT_END: ライン到達前は中央、到達時に一度だけ左旋回、その後は右端追従
        汎用状態dict(self._state)で管理する。
        offset_y_for_turn: この動作専用のライン到達判定Y座標（デフォルト400）
        Returns: (target_x, (left_speed, right_speed), ret_mode)
        """
        state = self._state.setdefault("turn_at_end", {"reached": False, "turned": False})
        x1, y1, x2, y2 = ROI_CNN
        # get_line_trace_edges_at_x320でラインがoffset_y_for_turnに到達しているか判定
        y_hit = get_line_trace_edges_at_x320(frame)
        reached = y_hit is not None and y_hit >= offset_y_for_turn
        left_speed = right_speed = None
        target_x = None
        if not state["reached"] and reached:
            state["reached"] = True
            state["turned"] = False
        if not state["reached"]:
            # ライン到達前は中央
            target_x = (x1 + x2) // 2
            return target_x, (left_speed, right_speed), None
        elif not state["turned"]:
            # 到達した瞬間に一度だけ左旋回
            result = self.trun_left()
            if result is not None:
                _, speeds, _ = result
                if speeds is not None:
                    left_speed, right_speed = speeds
                else:
                    left_speed, right_speed = 0, 0
            else:
                left_speed, right_speed = 0, 0
            state["turned"] = True
            return target_x, (left_speed, right_speed), None
        else:
            # 以降は右端追従はrun_manual.py側の通常ロジックに任せる
            # 状態リセットもここで行う
            self._state["turn_at_end"] = {"reached": False, "turned": False}
            return None, None, Mode.FOLLOW_RIGHT_EDGE