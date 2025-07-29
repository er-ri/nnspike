import time
from typing import Optional, Tuple

import numpy as np

from nnspike.constants import OFFSET_Y, ROI_CNN, Mode, BASE_SPEED
from nnspike.unit.etrobot import ETRobot
from nnspike.utils.control import (
    find_bottle_center,
    get_line_edges_at_y,
    get_line_trace_edges_at_x320,
    get_virtual_line_edges_at_y,
    find_blue_target_center,
    get_is_blue_line_at_y,
)


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
        以下の順で動作する:
        0. 左旋回 (左:40, 右:70, 0.8秒)
        1. 右旋回 (左:80, 右:50, 1.3秒)
        2. コースに応じて左端または右端追従モードへ復帰
        """
        state = self._state.setdefault("avoid_obstacle", {
            "phase": 0,
            "phase_start_time": None,
        })
        now = time.time()

        # 0. 左旋回（0.8秒）
        if state["phase"] == 0:
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            if now - state["phase_start_time"] < 0.8:
                left_speed, right_speed = 40, 70  # 左旋回
                return None, (left_speed, right_speed), Mode.AVOID_OBSTACLE
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 右旋回（1.3秒）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 1.3:
                left_speed, right_speed = 80, 50  # 右旋回
                return None, (left_speed, right_speed), Mode.AVOID_OBSTACLE
            else:
                state["phase"] = 2
                state["phase_start_time"] = now

        # 2. チェーン終了でリセット
        if state["phase"] == 2:
            # 状態リセット
            self._state["avoid_obstacle"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.FOLLOW_RIGHT_EDGE

    def carry_bottle1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 赤ボトル中心追従（red_pixel_countが一度3000以上となった時刻をred_detected_timeとし、red_detected_timeから5.2秒後に次段階へ遷移）
        1. 左旋回（0.8秒, 左:0, 右:60）
        2. 直進（1.0秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        3. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x, preference='left'）
        4. 直進（3.5秒, 両輪BASE_SPEED）
        5. 左旋回（0.8秒, 左:0, 右:60）
        6. 青検出（blue_pixel_countが一度1000以上→500以下になってから1秒後、または最大4秒でBACK_AND_TURN1に遷移）
        """
        state = self._state.setdefault("carry_bottle1", {
            "phase": 0,
            "phase_start_time": None,
            "red_detected_time": None,
            "pre_target_x": None,
            "eye_blue_start": None,
            "blue_over1000": False,
            "blue_under500_time": None,
        })
        now = time.time()
        # 0. 赤ボトル中心追従（5.2秒）
        center, _, red_pixel_count = find_bottle_center(image=image, color="red")
        if state["phase"] == 0:
            # 0. 赤ボトル中心追従（5.2秒）
            if state["red_detected_time"] is None:
                if red_pixel_count > 3000:
                    state["red_detected_time"] = now
                    state["phase_start_time"] = now
            if state["red_detected_time"] is None or (now - state["red_detected_time"] < 5.0):
                if red_pixel_count > 3000 and center is not None:
                    target_x = center[0]
                else:
                    left_x, _, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
                    x1, _, x2, _ = ROI_CNN
                    target_x = left_x if left_x is not None else (x1 + x2) // 2
                return target_x, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 左旋回（0.8秒）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 0.8:
                return None, (0, 60), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 2
                state["phase_start_time"] = now

        # 2. 直進（1.0秒, pre_target_xも中央にリセット）
        if state["phase"] == 2:
            if now - state["phase_start_time"] < 1.0:
                x1, _, x2, _ = ROI_CNN
                state["pre_target_x"] = (x1 + x2) // 2
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 3
                state["phase_start_time"] = now

        # 3. 仮想ライン直進（2.0秒）
        if state["phase"] == 3:
            if now - state["phase_start_time"] < 2.0:
                pre_target_x = state.get("pre_target_x")
                temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x)
                x1, _, x2, _ = ROI_CNN
                if temp_x is not None:
                    target_x = temp_x
                    state["pre_target_x"] = temp_x
                elif pre_target_x is not None:
                    target_x = pre_target_x
                else:
                    target_x = (x1 + x2) // 2
                    state["pre_target_x"] = target_x
                return target_x, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 4
                state["phase_start_time"] = now

        # 4. 直進（3.5秒）
        if state["phase"] == 4:
            if now - state["phase_start_time"] < 4.8:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 5
                state["phase_start_time"] = now

        # 5. 左旋回（0.8秒）
        if state["phase"] == 5:
            if now - state["phase_start_time"] < 0.8:
                return None, (0, 60), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 6
                state["phase_start_time"] = now

        # 6. 青検出（blue_pixel_countが一度1000以上→500以下になってから1秒後、または最大4秒でBACK_AND_TURN1に遷移）
        if state["phase"] == 6:
            if state['eye_blue_start'] is None:
                state['eye_blue_start'] = now
            if 'blue_over1000' not in state:
                state['blue_over1000'] = False
            if 'blue_under500_time' not in state:
                state['blue_under500_time'] = None


            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if not state['blue_over1000'] and blue_pixel_count > 1000:
                state['blue_over1000'] = True
            if state['blue_over1000'] and state['blue_under500_time'] is None and blue_pixel_count <= 500:
                state['blue_under500_time'] = now

            to_back_and_turn1 = False
            if state['blue_under500_time'] is not None:
                if now - state['blue_under500_time'] >= 1.5:
                    to_back_and_turn1 = True
            if now - state['eye_blue_start'] >= 8.0:
                to_back_and_turn1 = True

            if not to_back_and_turn1:
                x1, _, x2, _ = ROI_CNN
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (x1 + x2) // 2
                return target_x, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["pre_target_x"] = None
                state['eye_blue_start'] = None
                state['blue_over1000'] = False
                state['blue_under500_time'] = None
                state["phase"] = 0
                state["phase_start_time"] = None
                state["red_detected_time"] = None
                return None, None, Mode.BACK_AND_TURN1

    def back_and_turn1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 後退（1.8秒, 両輪BASE_SPEED）
        1. 左旋回（2.0秒, 左:0, 右:BASE_SPEED）
        2. 終了後CARRY_BOTTLE2へ遷移

        """
        state = self._state.setdefault("back_and_turn1", {
            "phase": 0,
            "phase_start_time": None,
        })
        now = time.time()

        # 0. 後退（1.8秒）
        if state["phase"] == 0:
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            if now - state["phase_start_time"] < 1.8:
                left_speed = right_speed = BASE_SPEED
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN1
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 左旋回（2.0秒）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 2.0:
                left_speed, right_speed = 0, BASE_SPEED
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN1
            else:
                state["phase"] = 2
                state["phase_start_time"] = now

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn1"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.CARRY_BOTTLE2

    def carry_bottle2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 青ボトル中心追従（blue_pixel_countが一度3000以上となったら中心追従、3000以下になった場合0.5秒後に左旋回フェーズへ遷移）
        1. 左旋回（1.2秒, 左:0, 右:60）
        2. 直進（2.5秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        3. 左旋回（0.8秒, 左:0, 右:60）
        4. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x, preference='right'）
        5. 直進（2.0秒, 両輪BASE_SPEED）
        6. 左旋回（0.8秒, 左:0, 右:60）
        7. 青検出（blue_pixel_countが一度1000以上→500以下になってから1秒後、または最大3秒でBACK_AND_TURN2に遷移）

        """

        state = self._state.setdefault("carry_bottle2", {
            "phase": 0,
            "phase_start_time": None,
            "blue_detected": False,
            "blue_lost_time": None,
            "pre_target_x": None,
            "blue_detected_time": None,
            "eye_blue_start": None,
        })
        now = time.time()
        center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")

        # 0. FOLLOW_RIGHT_EDGE + find_bottle_center(blue)
        # blue_pixel_countが一度3000以上となったら中心追従、3000以下になった場合（1度3000以上になった後）0.5秒後に右旋回フェーズへ遷移
        if state["phase"] == 0:
            if not state["blue_detected"]:
                if blue_pixel_count > 3000:
                    state["blue_detected"] = True
                    state["blue_lost_time"] = None
            if state["blue_detected"] and state["blue_lost_time"] is None and blue_pixel_count <= 3000:
                state["blue_lost_time"] = now
                state["phase_start_time"] = now
            if not state["blue_detected"] or (state["blue_lost_time"] is not None and now - state["blue_lost_time"] < 0.5):
                if blue_pixel_count > 3000 and center is not None:
                    target_x = center[0]
                else:
                    _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
                    x1, _, x2, _ = ROI_CNN
                    target_x = right_x if right_x is not None else (x1 + x2) // 2
                return target_x, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            elif state["blue_lost_time"] is not None and now - state["blue_lost_time"] >= 0.5:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. TURN_LEFT (blue_lost_timeから0.5秒経過後、1.2秒左旋回)
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 1.2:
                left_speed, right_speed = 0, 60  # 左旋回
                return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 2
                state["phase_start_time"] = now
                state["blue_detected_time"] = now

        # 2. FORWARD (blue_detected_time+0〜2.5秒, 2.5秒直進)
        if state["phase"] == 2:
            if now - state["phase_start_time"] < 2.5:
                left_speed = right_speed = BASE_SPEED
                return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 3
                state["phase_start_time"] = now
                x1, _, x2, _ = ROI_CNN
                state["pre_target_x"] = (x1 + x2) // 2

        # 3. TURN_LEFT (0.8秒左旋回)
        if state["phase"] == 3:
            if now - state["phase_start_time"] < 0.8:
                left_speed, right_speed = 0, 60  # 左旋回
                return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 4
                state["phase_start_time"] = now

        # 4. GATE_PASS (2.0秒get_virtual_line_edges_at_yで直進)
        if state["phase"] == 4:
            if now - state["phase_start_time"] < 2.0:
                pre_target_x = state.get("pre_target_x")
                temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x)
                x1, _, x2, _ = ROI_CNN
                if temp_x is not None:
                    target_x = temp_x
                    state["pre_target_x"] = temp_x
                elif pre_target_x is not None:
                    target_x = pre_target_x
                else:
                    target_x = (x1 + x2) // 2
                    state["pre_target_x"] = target_x
                left_speed = right_speed = BASE_SPEED
                return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 5
                state["phase_start_time"] = now

        # 5. FORWARD2 (2.0秒直進)
        if state["phase"] == 5:
            if now - state["phase_start_time"] < 2.0:
                left_speed = right_speed = BASE_SPEED
                return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 6
                state["phase_start_time"] = now

        # 6. TURN_LEFT (0.8秒左旋回)
        if state["phase"] == 6:
            if now - state["phase_start_time"] < 0.8:
                left_speed, right_speed = 0, 60
                return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 7
                state["phase_start_time"] = now

        # 7. EYE_BLUE: 青検出（bottle1と同じロジック、最大3秒）
        if state["phase"] == 7:
            if state['eye_blue_start'] is None:
                state['eye_blue_start'] = now
            if 'blue_over1000' not in state:
                state['blue_over1000'] = False
            if 'blue_under500_time' not in state:
                state['blue_under500_time'] = None

            center, _, blue_pixel_count = find_blue_target_center(image)
            if not state['blue_over1000'] and blue_pixel_count > 1000:
                state['blue_over1000'] = True
            if state['blue_over1000'] and state['blue_under500_time'] is None and blue_pixel_count <= 500:
                state['blue_under500_time'] = now

            to_back_and_turn2 = False
            if state['blue_under500_time'] is not None:
                if now - state['blue_under500_time'] >= 1.0:
                    to_back_and_turn2 = True
            if now - state['eye_blue_start'] >= 3.0:
                to_back_and_turn2 = True

            if not to_back_and_turn2:
                x1, _, x2, _ = ROI_CNN
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (x1 + x2) // 2
                left_speed = right_speed = BASE_SPEED
                return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                state["pre_target_x"] = None
                state['eye_blue_start'] = None
                state['blue_over1000'] = False
                state['blue_under500_time'] = None
                state["phase"] = 0
                state["phase_start_time"] = None
                state["blue_detected"] = False
                state["blue_lost_time"] = None
                state["blue_detected_time"] = None
                return None, None, Mode.BACK_AND_TURN2

    def back_and_turn2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 2秒間後退（両輪BASE_SPEED）
        1. 0.8秒右旋回（左:BASE_SPEED, 右:0）
        2. 終了後HEAD_GOALへ遷移

        """
        state = self._state.setdefault("back_and_turn2", {
            "phase": 0,
            "phase_start_time": None,
        })
        now = time.time()

        # 0. 2秒間後退
        if state["phase"] == 0:
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            if now - state["phase_start_time"] < 2.0:
                left_speed = right_speed = BASE_SPEED
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN2
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 0.8秒右旋回
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 0.8:
                left_speed, right_speed = BASE_SPEED, 0
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN2
            else:
                state["phase"] = 2
                state["phase_start_time"] = now

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn2"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.HEAD_GOAL

    def heading_goal(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. ライン到達前は中央追従
        1. 到達時に左旋回（0.8秒, 左:0, 右:60）
        2. 右端追従（get_line_edges_at_yで右端追従, 右端追従中にget_is_blue_line_at_y(target_y=OFFSET_Y)がTrueになったら0.5秒後にPAUSE）

        """
        state = self._state.setdefault("heading_goal", {
            "phase": 0,
            "phase_start_time": None,
            "blue_line_detected_time": None,
        })
        x1, y1, x2, y2 = ROI_CNN
        y_hit = get_line_trace_edges_at_x320(image)
        reached = y_hit is not None and y_hit >= 450
        left_speed = right_speed = None
        target_x = None
        now = time.time()

        # 0. ライン到達前は中央追従
        if state["phase"] == 0:
            if reached:
                state["phase"] = 1
                state["phase_start_time"] = now
            else:
                target_x = (x1 + x2) // 2
                return target_x, (left_speed, right_speed), Mode.HEAD_GOAL

        # 1. 到達時に0.8秒左旋回
        if state["phase"] == 1:
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            if now - state["phase_start_time"] < 0.8:
                left_speed, right_speed = 0, 60
                return target_x, (left_speed, right_speed), Mode.HEAD_GOAL
            else:
                state["phase"] = 2
                state["phase_start_time"] = now
                state["blue_line_detected_time"] = None

        # 2. 以降は右端追従（青ライン検出で0.5秒後にPAUSE）
        if state["phase"] == 2:
            if "right_trace_start" not in state or state["right_trace_start"] is None:
                state["right_trace_start"] = now
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (x1 + x2) // 2
            left_speed = right_speed = BASE_SPEED

            # 青ライン検出ロジック
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                if state["blue_line_detected_time"] is None:
                    state["blue_line_detected_time"] = now
                elif now - state["blue_line_detected_time"] >= 1.0:
                    # 状態リセットしてPAUSEへ
                    state["phase"] = 0
                    state["phase_start_time"] = None
                    state["blue_line_detected_time"] = None
                    return None, None, Mode.PAUSE
            else:
                state["blue_line_detected_time"] = None

            return target_x, (left_speed, right_speed), Mode.HEAD_GOAL

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
        以下の順で動作する:
        0. ライン到達前は中央追従
        1. 到達時に一度だけ左旋回（trun_left, 0.8秒, 左:0, 右:60）
        2. 以降は右端追従（run_manual.py側の通常ロジックに任せる）
        汎用状態dict(self._state)で管理。
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
            # 状態リセットは行わず、右端追従を継続
            return None, None, Mode.FOLLOW_RIGHT_EDGE

    def trun_left_gyro(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        ジャイロz角度の累積変化量（積分値）が-90度に達したらPAUSEに遷移する左旋回アクション。
        """
        state = self._state.setdefault("trun_left_gyro", {"last_gyro_z": None, "integrated_delta": 0.0})
        # 最新のgyro_z値を直接取得
        current_gyro_z = self.et.last_spike_status.sensors.gyro.z
        if state["last_gyro_z"] is None:
            state["last_gyro_z"] = current_gyro_z
        # 差分を積算
        delta = current_gyro_z - state["last_gyro_z"]
        state["integrated_delta"] += delta
        state["last_gyro_z"] = current_gyro_z
        # 左回転はzがマイナス方向に進む（累積-90で終了）
        if state["integrated_delta"] > -90:
            left_speed, right_speed = 0, 60
            return None, (left_speed, right_speed), Mode.TURN_LEFT_GYRO
        # 終了条件を満たしたら状態リセット
        state["last_gyro_z"] = None
        state["integrated_delta"] = 0.0
        return None, None, Mode.PAUSE

    def trun_right_gyro(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        ジャイロz角度の累積変化量（積分値）が+90度に達したらPAUSEに遷移する右旋回アクション。
        """
        state = self._state.setdefault("trun_right_gyro", {"last_gyro_z": None, "integrated_delta": 0.0})
        # 最新のgyro_z値を直接取得
        current_gyro_z = self.et.last_spike_status.sensors.gyro.z
        if state["last_gyro_z"] is None:
            state["last_gyro_z"] = current_gyro_z
        # 差分を積算
        delta = current_gyro_z - state["last_gyro_z"]
        state["integrated_delta"] += delta
        state["last_gyro_z"] = current_gyro_z
        # 右回転はzがプラス方向に進む（累積+90で終了）
        if state["integrated_delta"] < 90:
            left_speed, right_speed = 60, 0
            return None, (left_speed, right_speed), Mode.TURN_RIGHT_GYRO
        # 終了条件を満たしたら状態リセット
        state["last_gyro_z"] = None
        state["integrated_delta"] = 0.0
        return None, None, Mode.PAUSE