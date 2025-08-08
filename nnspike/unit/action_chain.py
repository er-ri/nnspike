
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
    is_x320_on_blue_target,
    is_x320_on_red_target,
    is_left_black_line_detected,
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
        # ROI_CNNをインスタンス変数に展開
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN

    def avoid_obstacle(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 左旋回 (左:40, 右:70, 0.8秒)
        1. 右旋回 (左:80, 右:50, 1.3秒)
        2. チェーン終了で右端追従モードへ復帰
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
                return None, (40, 70), Mode.AVOID_OBSTACLE
            state["phase"] = 1
            state["phase_start_time"] = now

        # 1. 右旋回（1.3秒）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 1.3:
                return None, (80, 50), Mode.AVOID_OBSTACLE
            state["phase"] = 2
            state["phase_start_time"] = now

        # 2. チェーン終了でリセット
        if state["phase"] == 2:
            self._state["avoid_obstacle"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.FOLLOW_RIGHT_EDGE

    def carry_bottle1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 右エッジトレース（赤ピクセル数が3000を超えたらphase1へ）
        1. 赤ボトル中心追従（3.0秒、赤が見えなければ中央）
        2. 直進（2.0秒）
        3. 左旋回（1.5秒, 左:0, 右:30）
        4. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x）
        5. 直進（4.8秒）
        6. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最大2秒）
        7. 青検出（1000超えたらphase8へ）
        8. 青1000以上の間center追従、500以下でphase9へ
        9. 青500以下になってから1秒間center追従、その後BACK_AND_TURN1
        """
        state = self._state.setdefault("carry_bottle1", {
            "phase": 0,
            "phase_start_time": None,
            "pre_target_x": None,
        })
        now = time.time()

        # 0. 右エッジトレース→赤3000超でphase1へ
        if state["phase"] == 0:
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            _, _, red_pixel_count = find_bottle_center(image=image, color="red")
            if red_pixel_count > 3000:
                state["phase"] = 1
                state["phase_start_time"] = now
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 1. 赤ボトル中心追従（3.0秒、赤が見えなければ中央）
        if state["phase"] == 1:
            center, _, _ = find_bottle_center(image=image, color="red")
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            if now - state["phase_start_time"] < 3.0:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE1
            state["phase"] = 2
            state["phase_start_time"] = now

        # 2. 右エッジトレース（target_y=300, 2.2秒）
        if state["phase"] == 2:
            if now - state["phase_start_time"] < 2.2:
                _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=300, threshold_value=80)
                target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE1
            state["phase"] = 3
            state["phase_start_time"] = now

        # 3. 左旋回（1.5秒, 左:0, 右:30）
        if state["phase"] == 3:
            if now - state["phase_start_time"] < 1.5:
                return None, (0, 30), Mode.CARRY_BOTTLE1
            state["phase"] = 4
            state["phase_start_time"] = now
            state["pre_target_x"] = (self.x1 + self.x2) // 2

        # 4. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x）
        if state["phase"] == 4:
            if now - state["phase_start_time"] < 2.0:
                pre_target_x = state.get("pre_target_x")
                temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    state["pre_target_x"] = temp_x
                elif pre_target_x is not None:
                    target_x = pre_target_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    state["pre_target_x"] = target_x
                return target_x, None, Mode.CARRY_BOTTLE1
            state["phase"] = 5
            state["phase_start_time"] = now

        # 5. 直進（5.0秒）
        if state["phase"] == 5:
            if now - state["phase_start_time"] < 5.0:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            state["phase"] = 6
            state["phase_start_time"] = now

        # 6. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最大2秒）
        if state["phase"] == 6:
            if (not is_x320_on_blue_target(image, x_tolerance=40)) and (now - state["phase_start_time"] < 2.0):
                return None, (0, 30), Mode.CARRY_BOTTLE1
            state["phase"] = 7
            state["phase_start_time"] = now

        # 7. 青検出（1000超えたらphase8へ）
        if state["phase"] == 7:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                state["phase"] = 8
                state["phase_start_time"] = now
            return target_x, None, Mode.CARRY_BOTTLE1

        # 8. 青1000以上の間center追従、500以下でphase9へ
        if state["phase"] == 8:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                state["phase"] = 9
                state["phase_start_time"] = now
            return target_x, None, Mode.CARRY_BOTTLE1

        # 9. 青500以下になってから0.8秒間center追従、その後BACK_AND_TURN1
        if state["phase"] == 9:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if now - state["phase_start_time"] < 0.8:
                return target_x, None, Mode.CARRY_BOTTLE1
            # 状態リセット
            self._state["carry_bottle1"] = {"phase": 0, "phase_start_time": None, "pre_target_x": None}
            return None, None, Mode.BACK_AND_TURN1

    def back_and_turn1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 後退（1.8秒, 両輪BASE_SPEED）
        1. 左旋回（is_x320_on_red_target(image, x_tolerance=40)がTrueになるまで、または最大3.0秒, 左:0, 右:30）
        2. 終了後CARRY_BOTTLE2へ遷移（状態リセット）
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
                return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN1
            state["phase"] = 1
            state["phase_start_time"] = now

        # 1. 左旋回（is_x320_on_red_target(image, x_tolerance=20)がTrueでも最低1.5秒は旋回、その後Trueなら即終了、最大3.0秒, 左:0, 右:30）
        if state["phase"] == 1:
            elapsed = now - state["phase_start_time"]
            if elapsed < 1.5:
                # 最低1.5秒は必ず旋回
                return None, (0, 30), Mode.BACK_AND_TURN1
            if (not is_x320_on_red_target(image, x_tolerance=20)) and (elapsed < 3.0):
                return None, (0, 30), Mode.BACK_AND_TURN1
            state["phase"] = 2
            state["phase_start_time"] = now

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn1"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.CARRY_BOTTLE2

    def carry_bottle2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 右エッジトレース（青ピクセル数が3000を超えたらphase1へ）
        1. 青ボトル中心追従（3000以上の間center追従、3000以下でphase2へ）
        2. 3000以下になってから0.5秒間center追従。その後phase3（左旋回is_left_black_line_detected(image) or 3秒）
        3. 左旋回（is_left_black_line_detected(image)がTrueになるまで、または3秒未満, 左:0, 右:30）
        4. 直進（2.5秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        5. 左旋回（1.5秒, 左:0, 右:30）
        6. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x）
        7. 直進（2.0秒, 両輪BASE_SPEED）
        8. 左旋回（is_x320_on_blue_target(image, x_tolerance=40)がTrueになるまで、または2秒未満, 左:0, 右:30）
        9. 青検出（青ピクセル数が1000を超えたらphase10へ）
        10. 青ピクセルが500以下まで減るまでcenter追従（500以下でphase11へ）
        11. 500以下になってから1秒間center追従、その後BACK_AND_TURN2へ遷移
        """
        state = self._state.setdefault("carry_bottle2", {
            "phase": 0,
            "phase_start_time": None,
            "pre_target_x": None,
        })
        now = time.time()

        # 0. 青ピクセル数が3000を超える前は単純直進、超えたらphase1へ
        if state["phase"] == 0:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if blue_pixel_count > 3000:
                state["phase"] = 1
                state["phase_start_time"] = now
                target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2

        # 1. 青ボトル中心追従（3000以上の間center追従、3000以下になってから0.2秒間center追従、その後phase2へ）
        if state["phase"] == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count > 3000:
                state["below3000_time"] = None
                target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            # 3000以下になった瞬間の時刻を記録
            if "below3000_time" not in state or state["below3000_time"] is None:
                state["below3000_time"] = now
            # 0.2秒間はcenter追従を継続（centerがNoneなら中央）
            if now - state["below3000_time"] < 0.2:
                target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            # 0.2秒経過したらphase2へ
            state["phase"] = 2
            state["phase_start_time"] = now
            state["below3000_time"] = None

        # 2. 3000以下になってから0.3秒間center追従。その後phase3（左旋回is_left_black_line_detected(image) or 3秒）
        if state["phase"] == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            elapsed = now - state["phase_start_time"]
            if elapsed < 0.3:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            state["phase"] = 3
            state["phase_start_time"] = now

        # 3. 左旋回（is_left_black_line_detected(image)がTrueになるまで、または3秒未満, 左:0, 右:30）
        if state["phase"] == 3:
            if (not is_left_black_line_detected(image)) and (now - state["phase_start_time"] < 3.0):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 4
            state["phase_start_time"] = now

        # 4. 直進（2.3秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        if state["phase"] == 4:
            if now - state["phase_start_time"] < 2.3:
                state["pre_target_x"] = (self.x1 + self.x2) // 2
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 5
            state["phase_start_time"] = now

        # 5. 左旋回（1.3秒, 左:0, 右:30）
        if state["phase"] == 5:
            if now - state["phase_start_time"] < 1.3:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 6
            state["phase_start_time"] = now

        # 6. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x）
        if state["phase"] == 6:
            if now - state["phase_start_time"] < 2.0:
                pre_target_x = state.get("pre_target_x")
                temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    state["pre_target_x"] = temp_x
                elif pre_target_x is not None:
                    target_x = pre_target_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    state["pre_target_x"] = target_x
                return target_x, None, Mode.CARRY_BOTTLE2
            state["phase"] = 7
            state["phase_start_time"] = now

        # 7. 直進（2.3秒, 両輪BASE_SPEED）
        if state["phase"] == 7:
            if now - state["phase_start_time"] < 2.3:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 8
            state["phase_start_time"] = now

        # 8. 左旋回（is_x320_on_blue_target(image, x_tolerance=40)がTrueでも最低1.2秒は旋回、その後Trueなら即終了、最大2秒, 左:0, 右:30）
        if state["phase"] == 8:
            elapsed = now - state["phase_start_time"]
            if elapsed < 1.2:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            if (not is_x320_on_blue_target(image, x_tolerance=40)) and (elapsed < 2.0):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 9
            state["phase_start_time"] = now

        # 9. 青検出（青ピクセル数が1000を超えたらphase10へ、または最大2秒でphase10へ）
        if state["phase"] == 9:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            elapsed = now - state["phase_start_time"]
            if blue_pixel_count > 1000 or elapsed >= 2.0:
                state["phase"] = 10
                state["phase_start_time"] = now
            return target_x, None, Mode.CARRY_BOTTLE2

        # 10. 青ピクセルが500以下まで減るまでcenter追従（500以下でphase11へ、または最大2秒でphase11へ）
        if state["phase"] == 10:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            elapsed = now - state["phase_start_time"]
            if blue_pixel_count <= 500 or elapsed >= 2.0:
                state["phase"] = 11
                state["phase_start_time"] = now
            return target_x, None, Mode.CARRY_BOTTLE2

        # 11. 500以下になってから0.8秒間center追従、その後BACK_AND_TURN2へ遷移
        if state["phase"] == 11:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if now - state["phase_start_time"] < 0.8:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            # 状態リセット
            self._state["carry_bottle2"] = {"phase": 0, "phase_start_time": None, "pre_target_x": None}
            return None, None, Mode.BACK_AND_TURN2

    def back_and_turn2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. 2.0秒間後退（両輪BASE_SPEED）
        1. 1.5秒右旋回（左:30, 右:0）
        2. 終了後HEAD_GOALへ遷移（状態リセット）
        """
        state = self._state.setdefault("back_and_turn2", {
            "phase": 0,
            "phase_start_time": None,
        })
        now = time.time()

        # 0. 1.5秒間後退
        if state["phase"] == 0:
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            if now - state["phase_start_time"] < 1.5:
                return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN2
            state["phase"] = 1
            state["phase_start_time"] = now

        # 1. 1.3秒右旋回（左:30, 右:0）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 1.3:
                return None, (30, 0), Mode.BACK_AND_TURN2
            state["phase"] = 2
            state["phase_start_time"] = now

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn2"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.HEAD_GOAL

    def heading_goal(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        0. ライン到達前は中央追従（y_hit >= 450）
        1. 到達直後0.5秒は直進
        2. 左旋回（1.5秒, 左:0, 右:30）
        3. 右エッジトレース（青ライン検出でphase4へ）
        4. 青ライン検出後、1.5秒右エッジトレースしたらPAUSE（状態リセット）
        """
        state = self._state.setdefault("heading_goal", {
            "phase": 0,
            "phase_start_time": None,
        })
        now = time.time()

        # 0. ライン到達前は中央追従（y_hit >= 450）、ただし最長1秒で打ち切り
        if state["phase"] == 0:
            if state["phase_start_time"] is None:
                state["phase_start_time"] = now
            y_hit = get_line_trace_edges_at_x320(image)
            elapsed = now - state["phase_start_time"]
            if (y_hit is not None and y_hit >= 450) or (elapsed >= 1.0):
                state["phase"] = 1
                state["phase_start_time"] = now
            else:
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 1. 到達直後0.5秒は直進
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 0.5:
                target_x = (self.x1 + self.x2) // 2
                return target_x, (BASE_SPEED, BASE_SPEED), Mode.HEAD_GOAL
            state["phase"] = 2
            state["phase_start_time"] = now

        # 2. 左旋回（1.5秒, 左:0, 右:30）
        if state["phase"] == 2:
            if now - state["phase_start_time"] < 1.5:
                return None, (0, 30), Mode.HEAD_GOAL
            state["phase"] = 3
            state["phase_start_time"] = now

        # 3. 右エッジトレース（青ライン検出でphase4へ）
        if state["phase"] == 3:
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (self.x1 + self.x2) // 2
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                state["phase"] = 4
                state["phase_start_time"] = now
                return target_x, None, Mode.HEAD_GOAL
            return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、1.5秒右エッジトレースしたらPAUSE（状態リセット）
        if state["phase"] == 4:
            if now - state["phase_start_time"] >= 1.5:
                self._state["heading_goal"] = {"phase": 0, "phase_start_time": None}
                return None, None, Mode.PAUSE
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (self.x1 + self.x2) // 2
            return target_x, None, Mode.HEAD_GOAL

    def trun_left(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        左旋回アクションを実行する（backup/actions.pyのturn_left相当）。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 1.5:
            return None, (0, 30), Mode.TURN_LEFT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def trun_right(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        右旋回アクションを実行する（backup/actions.pyのturn_right相当）。
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 1.5:
            return None, (30, 0), Mode.TURN_RIGHT
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
            return None, (0, 50), Mode.SMALL_TURN_LEFT
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
            return None, (50, 0), Mode.SMALL_TURN_RIGHT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def blue_bottle_catch(self, image: np.ndarray) -> tuple:
        """
        ブルーボトルキャッチモード: 青重心に向かう。3000ピクセル超で検知、3000未満で0.8秒直進、その後PAUSE。
        戻り値: (target_x, (left_speed, right_speed), mode)
        """
        state = self._state.setdefault("blue_bottle_catch", {"detected": False, "below3000_time": None})
        blue_cx, _, blue_pixel_count = find_bottle_center(image, color="blue")
        now = time.time()
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        BASE_SPEED = 45  # run_manual.pyと合わせる
        target_x = None
        left_speed = right_speed = None
        mode = Mode.BLUE_BOTTLE_CATCH
        if not state["detected"]:
            if blue_pixel_count > 3000:
                state["detected"] = True
            state["below3000_time"] = None
            if blue_cx is not None:
                target_x = blue_cx[0]
            else:
                target_x = (x1 + x2) // 2
            return target_x, None, mode
        else:
            if blue_pixel_count < 3000:
                if state.get("below3000_time") is None:
                    state["below3000_time"] = now
                # 0.8秒間直進
                if now - state["below3000_time"] < 0.8:
                    target_x = (x1 + x2) // 2
                    left_speed = right_speed = BASE_SPEED
                    return target_x, (left_speed, right_speed), mode
                else:
                    # 状態リセットしPAUSEへ
                    state["detected"] = False
                    state["below3000_time"] = None
                    target_x = (x1 + x2) // 2
                    return target_x, (0, 0), Mode.PAUSE
            else:
                state["below3000_time"] = None
                if blue_cx is not None:
                    target_x = blue_cx[0]
                else:
                    target_x = (x1 + x2) // 2
                return target_x, None, mode

    def turn_at_end(self, frame, offset_y_for_turn=400):
        """
        以下の順で動作する:
        0. ライン到達前は中央追従
        1. 到達直後0.5秒は直進
        2. 一度だけ左旋回（trun_left, 0.8秒, 左:0, 右:60）
        3. 以降は右端追従（run_manual.py側の通常ロジックに任せる）
        汎用状態dict(self._state)で管理。
        offset_y_for_turn: この動作専用のライン到達判定Y座標（デフォルト400）
        Returns: (target_x, (left_speed, right_speed), ret_mode)
        """
        state = self._state.setdefault("turn_at_end", {"phase": 0, "phase_start_time": None})
        now = time.time()
        y_hit = get_line_trace_edges_at_x320(frame)

        # 0. ライン到達前は中央追従
        if state["phase"] == 0:
            if y_hit is not None and y_hit >= offset_y_for_turn:
                state["phase"] = 1
                state["phase_start_time"] = now
            else:
                target_x = (self.x1 + self.x2) // 2
                return target_x, (None, None), None

        # 1. 到達直後0.5秒は直進
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 0.5:
                target_x = (self.x1 + self.x2) // 2
                return target_x, (BASE_SPEED, BASE_SPEED), None
            state["phase"] = 2
            state["phase_start_time"] = now

        # 2. 一度だけ左旋回（0.8秒, 左:0, 右:60）
        if state["phase"] == 2:
            if now - state["phase_start_time"] < 0.8:
                return None, (0, 60), None
            state["phase"] = 3

        # 3. 以降は右端追従（run_manual.py側の通常ロジックに任せる）
        if state["phase"] == 3:
            return None, None, Mode.FOLLOW_RIGHT_EDGE

    def turn_left_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        左旋回（右モーターBの相対位置差分で判定、430未満の間は左:0,右:30で継続。430超えたらPAUSE）
        et.get_spike_status().motors["B"].relative_position, et.get_spike_status().motors["A"].relative_positionを条件として利用
        """
        status = self.et.get_spike_status()
        # 右モーターの初期位置を記録
        if not hasattr(self, '_right_position_start') or self._right_position_start is None:
            if status is not None and status.motors.get("B") is not None:
                self._right_position_start = status.motors.get("B").relative_position
            else:
                self._right_position_start = None

        if status is not None:
            right_position = status.motors.get("B").relative_position if status.motors.get("B") is not None else None
            # 右(B)の開始～現在の差分が430を超えたら停止
            if self._right_position_start is not None and right_position is not None:
                if abs(right_position - self._right_position_start) > 430:
                    self._right_position_start = None
                    return None, None, Mode.PAUSE
                else:
                    return None, (0, 30), Mode.TURN_LEFT_RELATIVE

        self._right_position_start = None
        return None, None, Mode.PAUSE

    def turn_right_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        右旋回（左モーターAの相対位置差分で判定、430未満の間は左:30,右:0で継続。430超えたらPAUSE）
        et.get_spike_status().motors["A"].relative_position, et.get_spike_status().motors["B"].relative_positionを条件として利用
        """
        status = self.et.get_spike_status()
        # 左モーターの初期位置を記録
        if not hasattr(self, '_left_position_start') or self._left_position_start is None:
            if status is not None and status.motors.get("A") is not None:
                self._left_position_start = status.motors.get("A").relative_position
            else:
                self._left_position_start = None

        if status is not None:
            left_position = status.motors.get("A").relative_position if status.motors.get("A") is not None else None
            # 左(A)の開始～現在の差分が430を超えたら停止
            if self._left_position_start is not None and left_position is not None:
                if abs(left_position - self._left_position_start) > 430:
                    self._left_position_start = None
                    return None, None, Mode.PAUSE
                else:
                    return None, (30, 0), Mode.TURN_RIGHT_RELATIVE

        self._left_position_start = None
        return None, None, Mode.PAUSE



# --- 以下、*_relativeメソッド（元メソッド完全コピー） ---

    def carry_bottle1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        carry_bottle1の位置判定バージョン。
        以下の順で動作する:
        0. 右エッジトレース（赤ピクセル数が3000を超えたらphase1へ）
        1. 赤ボトル中心追従（右モーター1100ユニット移動まで、赤が見えなければ中央）
        2. 右エッジトレース（右モーター1100ユニット移動まで）
        3. 左旋回（右モーター450ユニット移動まで, 左:0, 右:30）
        4. 仮想ライン直進（右モーター1000ユニット移動まで, get_virtual_line_edges_at_y, previous_center_x=pre_target_x）
        5. 直進（右モーター1900ユニット移動まで）
        6. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最大右モーター500ユニット）
        7. 青検出（1000超えたらphase8へ）
        8. 青1000以上の間center追従、500以下でphase9へ
        9. 青500以下になってから右モーター100ユニット移動まで center追従、その後BACK_AND_TURN1
        """
        state = self._state.setdefault("carry_bottle1_relative", {
            "phase": 0,
            "pre_target_x": None,
            "right_position_start": None,
        })
        status = self.et.get_spike_status()

        # 0. 右エッジトレース→赤3000超でphase1へ
        if state["phase"] == 0:
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            _, _, red_pixel_count = find_bottle_center(image=image, color="red")
            if red_pixel_count > 2000:
                state["phase"] = 1
                # phase1用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 1. 赤ボトル中心追従（右モーター相対位置差分が500未満の間、赤が見えなければ中央）
        if state["phase"] == 1:
            center, _, _ = find_bottle_center(image=image, color="red")
            red_result = find_bottle_center(image, color='red')
            red_px = red_result[2] if red_result else None
            # 右モーター相対位置差分で継続判定
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 500:
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 1100超えたら次フェーズへ
            state["phase"] = 2
            # phase2用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 2. 右エッジトレース（右モーター相対位置差分が1600未満の間）
        if state["phase"] == 2:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 1600:
                    _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=300, threshold_value=80)
                    target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 1600超えたら次フェーズへ
            state["phase"] = 3
            state["pre_target_x"] = (self.x1 + self.x2) // 2
            # phase3用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 3. 左旋回（右モーター相対位置差分が440未満の間 左:0, 右:30）
        if state["phase"] == 3:
            status = self.et.get_spike_status()
            if "right_position_start" not in state or state["right_position_start"] is None:
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 440:
                    return None, (0, 30), Mode.CARRY_BOTTLE1
            # 440超えたら次フェーズへ
            state["phase"] = 4
            state["pre_target_x"] = (self.x1 + self.x2) // 2
            # phase4用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 4. 仮想ライン直進（右モーター700ユニット移動まで, get_virtual_line_edges_at_y, previous_center_x=pre_target_x）
        if state["phase"] == 4:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 700:
                    pre_target_x = state.get("pre_target_x")
                    temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x)
                    if temp_x is not None:
                        target_x = temp_x
                        state["pre_target_x"] = temp_x
                    elif pre_target_x is not None:
                        target_x = pre_target_x
                    else:
                        target_x = (self.x1 + self.x2) // 2
                        state["pre_target_x"] = target_x
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 700超えたら次フェーズへ
            state["phase"] = 5
            # phase5用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 5. 直進（右モーター2300ユニット移動まで）
        if state["phase"] == 5:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 2300:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            # 2300超えたら次フェーズへ
            state["phase"] = 6
            # phase6用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 6. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最大右モーター500ユニット）
        if state["phase"] == 6:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=50)
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 500
            
            if (not blue_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE1
            state["phase"] = 7

        # 7. 青検出（1000超えたらphase8へ）
        if state["phase"] == 7:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                state["phase"] = 8
            return target_x, None, Mode.CARRY_BOTTLE1

        # 8. 青1000以上の間center追従、500以下でphase9へ
        if state["phase"] == 8:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                state["phase"] = 9
                # phase9用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            return target_x, None, Mode.CARRY_BOTTLE1

        # 9. 青500以下になってから右モーター100ユニット移動まで center追従、その後BACK_AND_TURN1
        if state["phase"] == 9:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            # 右モーター位置差分で継続判定
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 300:
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 状態リセット
            self._state["carry_bottle1_relative"] = {
                "phase": 0, 
                "pre_target_x": None,
                "right_position_start": None,
            }
            return None, None, Mode.BACK_AND_TURN1

    def back_and_turn1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        back_and_turn1の位置判定バージョン。
        以下の順で動作する:
        0. 後退（右モーター600ユニット移動まで, 両輪BASE_SPEED）
        1. 左旋回（is_x320_on_red_target(image, x_tolerance=20)検出まで、最大右モーター1000ユニット, 左:0, 右:30）
        2. 終了後CARRY_BOTTLE2へ遷移（状態リセット）
        """
        state = self._state.setdefault("back_and_turn1_relative", {
            "phase": 0,
            "right_position_start": None,
        })
        status = self.et.get_spike_status()

        # 0. 後退（右モーター600ユニット移動まで）
        if state["phase"] == 0:
            if state["right_position_start"] is None:
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 600:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN1
            state["phase"] = 1
            # phase1用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 1. 左旋回（is_x320_on_red_target(image, x_tolerance=40)検出まで、最低右モーター450ユニット、最大右モーター950ユニット, 左:0, 右:30）
        if state["phase"] == 1:
            red_target_detected = is_x320_on_red_target(image, x_tolerance=40)
            position_limit_reached = False
            minimum_position_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_diff = abs(current_pos - state["right_position_start"])
                minimum_position_reached = position_diff >= 450
                position_limit_reached = position_diff >= 950
            
            # 最低450ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (0, 30), Mode.BACK_AND_TURN1
            # 450ユニット超えてから、ターゲット検出または950ユニット到達まで継続
            if (not red_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.BACK_AND_TURN1
            state["phase"] = 2

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn1_relative"] = {"phase": 0, "right_position_start": None}
            return None, None, Mode.CARRY_BOTTLE2

    def carry_bottle2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        carry_bottle2の位置判定バージョン。
        以下の順で動作する:
        0. 右エッジトレース（青ピクセル数が3000を超えたらphase1へ）
        1. 青ボトル中心追従（3000以上の間center追従、3000以下で右モーター100ユニット移動まで追従、その後phase2へ）
        2. 右モーター100ユニット移動まで center追従。その後phase3
        3. 左旋回（is_left_black_line_detected(image)検出まで、最大右モーター1000ユニット, 左:0, 右:30）
        4. 直進（右モーター930ユニット移動まで, 両輪BASE_SPEED）
        5. 左旋回（右モーター380ユニット移動まで, 左:0, 右:30）
        6. 仮想ライン直進（右モーター800ユニット移動まで, get_virtual_line_edges_at_y）
        7. 直進（右モーター1000ユニット移動まで, 両輪BASE_SPEED）
        8. 左旋回（is_x320_on_blue_target検出まで、最大右モーター500ユニット, 左:0, 右:30）
        9. 青検出（青ピクセル数1000超えたらphase10へ、最大右モーター400ユニット）
        10. 青ピクセルが500以下まで減るまでcenter追従（500以下でphase11へ、最大右モーター400ユニット）
        11. 右モーター100ユニット移動まで center追従、その後BACK_AND_TURN2へ遷移
        """
        state = self._state.setdefault("carry_bottle2_relative", {
            "phase": 0,
            "pre_target_x": None,
            "right_position_start": None,
        })
        status = self.et.get_spike_status()

        # 0. 青ピクセル数が3000を超える前は単純直進、超えたらphase1へ
        if state["phase"] == 0:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if blue_pixel_count > 3000:
                state["phase"] = 1
                # phase1用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
                target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2

        # 1. 青ボトル中心追従（3000以上の間center追従、3000以下で右モーター100ユニット移動まで追従、その後phase2へ）
        if state["phase"] == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count > 3000:
                state["below3000_position_start"] = None
                target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            # 3000以下になった瞬間の位置を記録
            if "below3000_position_start" not in state or state["below3000_position_start"] is None:
                if status is not None and status.motors.get("B") is not None:
                    state["below3000_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["below3000_position_start"] = None
            # 右モーター100ユニット移動まで center追従を継続
            if status is not None and status.motors.get("B") is not None and state["below3000_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["below3000_position_start"]) < 100:
                    target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE2
            # 100ユニット移動したらphase2へ
            state["phase"] = 2
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None
            state["below3000_position_start"] = None

        # 2. 右モーター100ユニット移動まで center追従。その後phase3
        if state["phase"] == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 100:
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE2
            state["phase"] = 3
            # phase3用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 3. 左旋回（is_left_black_line_detected(image)検出まで、最大右モーター1000ユニット, 左:0, 右:30）
        if state["phase"] == 3:
            line_detected = is_left_black_line_detected(image)
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 1000
            
            if (not line_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 4
            # phase4用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 4. 直進（右モーター910ユニット移動まで, 両輪BASE_SPEED）
        if state["phase"] == 4:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 910:
                    state["pre_target_x"] = (self.x1 + self.x2) // 2
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 5
            # phase5用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 5. 左旋回（右モーター400ユニット移動まで, 左:0, 右:30）
        if state["phase"] == 5:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 400:
                    return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 6
            # phase6用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 6. 仮想ライン直進（右モーター800ユニット移動まで, get_virtual_line_edges_at_y）
        if state["phase"] == 6:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 800:
                    pre_target_x = state.get("pre_target_x")
                    temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x)
                    if temp_x is not None:
                        target_x = temp_x
                        state["pre_target_x"] = temp_x
                    elif pre_target_x is not None:
                        target_x = pre_target_x
                    else:
                        target_x = (self.x1 + self.x2) // 2
                        state["pre_target_x"] = target_x
                    return target_x, None, Mode.CARRY_BOTTLE2
            state["phase"] = 7
            # phase7用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 7. 直進（右モーター1000ユニット移動まで, 両輪BASE_SPEED）
        if state["phase"] == 7:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 1000:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 8
            # phase8用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 8. 左旋回（is_x320_on_blue_target検出まで、最低右モーター300ユニット、最大右モーター500ユニット, 左:0, 右:30）
        if state["phase"] == 8:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=50)
            position_limit_reached = False
            minimum_position_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_diff = abs(current_pos - state["right_position_start"])
                minimum_position_reached = position_diff >= 300
                position_limit_reached = position_diff >= 500
            
            # 最低300ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            # 300ユニット超えてから、ターゲット検出または500ユニット到達まで継続
            if (not blue_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 9
            # phase9用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 9. 青検出（青ピクセル数1000超えたらphase10へ、最大右モーター400ユニット）
        if state["phase"] == 9:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 400
            
            if blue_pixel_count > 1000 or position_limit_reached:
                state["phase"] = 10
                # phase10用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            return target_x, None, Mode.CARRY_BOTTLE2

        # 10. 青ピクセルが500以下まで減るまでcenter追従（500以下でphase11へ、最大右モーター400ユニット）
        if state["phase"] == 10:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 400
            
            if blue_pixel_count <= 500 or position_limit_reached:
                state["phase"] = 11
                # phase11用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            return target_x, None, Mode.CARRY_BOTTLE2

        # 11. 右モーター300ユニット移動まで center追従、その後BACK_AND_TURN2へ遷移
        if state["phase"] == 11:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 300:
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE2
            # 状態リセット
            self._state["carry_bottle2_relative"] = {
                "phase": 0, 
                "pre_target_x": None,
                "right_position_start": None,
            }
            return None, None, Mode.BACK_AND_TURN2

    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        back_and_turn2の位置判定バージョン（右旋回のため左モーター位置追跡）。
        以下の順で動作する:
        0. 左モーター570ユニット移動まで後退（両輪BASE_SPEED）
        1. 左モーター370ユニット移動まで右旋回（左:30, 右:0）
        2. 終了後HEAD_GOALへ遷移（状態リセット）
        """
        state = self._state.setdefault("back_and_turn2_relative", {
            "phase": 0,
            "left_position_start": None,
        })
        status = self.et.get_spike_status()

        # 0. 左モーター570ユニット移動まで後退
        if state["phase"] == 0:
            if state["left_position_start"] is None:
                if status is not None and status.motors.get("A") is not None:
                    state["left_position_start"] = abs(status.motors["A"].relative_position)
                else:
                    state["left_position_start"] = None

            if status is not None and status.motors.get("A") is not None and state["left_position_start"] is not None:
                current_pos = abs(status.motors["A"].relative_position)
                if abs(current_pos - state["left_position_start"]) < 570:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN2
            state["phase"] = 1
            # phase1用 左モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("A") is not None:
                state["left_position_start"] = abs(status.motors["A"].relative_position)
            else:
                state["left_position_start"] = None

        # 1. 左モーター350ユニット移動まで右旋回（左:30, 右:0）
        if state["phase"] == 1:
            if status is not None and status.motors.get("A") is not None and state["left_position_start"] is not None:
                current_pos = abs(status.motors["A"].relative_position)
                if abs(current_pos - state["left_position_start"]) < 350:
                    return None, (30, 0), Mode.BACK_AND_TURN2
            state["phase"] = 2

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn2_relative"] = {"phase": 0, "left_position_start": None}
            return None, None, Mode.HEAD_GOAL

    def heading_goal_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        heading_goalの位置判定バージョン（右モーター位置追跡）。
        以下の順で動作する:
        0. ライン到達前は中央追従（y_hit >= 450）、ただし最長右モーター700ユニットで打ち切り
        1. 到達直後右モーター100ユニット移動まで直進
        2. 左旋回（右モーター450ユニット移動まで, 左:0, 右:30）
        3. 右エッジトレース（青ライン検出でphase4へ）
        4. 青ライン検出後、右モーター300ユニット移動まで右エッジトレースしたらPAUSE（状態リセット）
        """
        state = self._state.setdefault("heading_goal_relative", {
            "phase": 0,
            "right_position_start": None,
        })
        status = self.et.get_spike_status()

        # 0. ライン到達前は中央追従（y_hit >= 450）、ただし最長右モーター700ユニットで打ち切り
        if state["phase"] == 0:
            if state["right_position_start"] is None:
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None

            y_hit = get_line_trace_edges_at_x320(image)
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 700
            
            if (y_hit is not None and y_hit >= 450) or position_limit_reached:
                state["phase"] = 1
                # phase1用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
            else:
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 1. 到達直後右モーター100ユニット移動まで直進
        if state["phase"] == 1:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 100:
                    target_x = (self.x1 + self.x2) // 2
                    return target_x, (BASE_SPEED, BASE_SPEED), Mode.HEAD_GOAL
            state["phase"] = 2
            # phase2用 右モーター相対位置記録（絶対値）
            if status is not None and status.motors.get("B") is not None:
                state["right_position_start"] = abs(status.motors["B"].relative_position)
            else:
                state["right_position_start"] = None

        # 2. 左旋回（右モーター450ユニット移動まで, 左:0, 右:30）
        if state["phase"] == 2:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                if abs(current_pos - state["right_position_start"]) < 450:
                    return None, (0, 30), Mode.HEAD_GOAL
            state["phase"] = 3

        # 3. 右エッジトレース（青ライン検出でphase4へ）
        if state["phase"] == 3:
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (self.x1 + self.x2) // 2
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                state["phase"] = 4
                # phase4用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position)
                else:
                    state["right_position_start"] = None
                return target_x, None, Mode.HEAD_GOAL
            return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、右モーター600ユニット移動まで右エッジトレースしたらPAUSE（状態リセット）
        if state["phase"] == 4:
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 600
            
            if position_limit_reached:
                self._state["heading_goal_relative"] = {"phase": 0, "right_position_start": None}
                return None, None, Mode.PAUSE
            
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (self.x1 + self.x2) // 2
            return target_x, None, Mode.HEAD_GOAL

    def nvidia_follow(self, image: np.ndarray, nvidia_mode_prediction: Optional[int]) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        NVIDIA_FOLLOWモードの処理を実行する。
        
        【挙動の流れ】
        1. 右モーター相対位置をチェック
           - abs(right_pos) > 21000: CARRY_BOTTLE1モードに自動切り替え
        
        2. abs(right_pos) >= 7000 かつ nvidia_mode_prediction が有効な場合:
           - NVIDIAモデルの予測に基づいて動作を選択
           - FOLLOW_LEFT_EDGE予測: 左エッジトレース処理を実行
           - それ以外（FOLLOW_RIGHT_EDGE含む全て）: 一律右エッジトレース処理を実行
        
        3. abs(right_pos) < 7000 または予測が無効な場合:
           - 通常の右エッジトレース処理を実行
        
        4. 障害物回避処理（両方のケースで共通）:
           - 黄色ピクセル数 > 14000 かつ abs(right_pos) <= 5000: 障害物回避開始
           - 左旋回0.8秒 → 右旋回1.3秒 → 通常動作復帰の3段階実行
           - 状態は "nvidia_avoid_obstacle" キーで管理
        
        5. 黄色ボトル追従:
           - 黄色ピクセル数 > 3000: 黄色重心に向かう
           - それ以外: ライン右エッジまたは中央に向かう
        
        Args:
            image: カメラからのフレーム画像
            nvidia_mode_prediction: NVIDIAモデルのモード予測結果 (Mode.FOLLOW_RIGHT_EDGE.value or Mode.FOLLOW_LEFT_EDGE.value)
            
        Returns:
            (target_x, (left_speed, right_speed), mode)
            - target_x: 目標X座標（ROI座標系）
            - speeds: 左右モーター速度のタプル（障害物回避時のみ指定、通常時はNone）
            - mode: 次のモード（通常はNVIDIA_FOLLOW維持、条件により CARRY_BOTTLE1 に切り替え）
        """
        # 右モーター位置を取得
        status = self.et.get_spike_status()
        right_pos = status.motors["B"].relative_position if status and status.motors.get("B") else None
        
        # 右相対距離が21000を超えたらCARRY_BOTTLE1に切り替え
        if right_pos is not None and abs(right_pos) > 21000:
            print(f"Right position {abs(right_pos)} exceeded 21000, switching to CARRY_BOTTLE1")
            target_x = (self.x1 + self.x2) // 2
            return target_x, None, Mode.CARRY_BOTTLE1
        
        # abs(right_pos) >= 7000かつ左エッジ予測の場合のみ左エッジトレース
        if right_pos is not None and abs(right_pos) >= 7000 and nvidia_mode_prediction == Mode.FOLLOW_LEFT_EDGE.value:
            # 左エッジモード予測の場合のみ：左エッジトレース処理
            left_x, _, _ = get_line_edges_at_y(image, ROI_CNN, OFFSET_Y, 80)
            if left_x is not None:
                target_x = left_x
            else:
                target_x = (self.x1 + self.x2) // 2
            return target_x, None, Mode.NVIDIA_FOLLOW
        
        # それ以外の全ての場合：右エッジトレース処理
        yellow_result = find_bottle_center(image, color="yellow")
        if yellow_result is not None:
            if len(yellow_result) == 3:
                yellow_cx, _, yellow_pixel_count = yellow_result
            else:
                yellow_cx, yellow_pixel_count = None, 0
        else:
            yellow_cx, yellow_pixel_count = None, 0
        _, right_x, _ = get_line_edges_at_y(image, ROI_CNN, OFFSET_Y, 80)
        
        # 障害物回避処理（right_pos <= 5000の場合のみ）
        if yellow_pixel_count > 14000 and yellow_cx is not None and (right_pos is None or abs(right_pos) <= 5000):
            # 障害物回避を直接実装
            state = self._state.setdefault("nvidia_avoid_obstacle", {
                "phase": 0,
                "phase_start_time": None,
            })
            now = time.time()
            
            # 0. 左旋回（0.8秒）
            if state["phase"] == 0:
                if state["phase_start_time"] is None:
                    state["phase_start_time"] = now
                if now - state["phase_start_time"] < 0.8:
                    print("Avoiding obstacle (right edge trace) - Phase 0: Left turn...")
                    target_x = (self.x1 + self.x2) // 2
                    return target_x, (40, 70), Mode.NVIDIA_FOLLOW
                state["phase"] = 1
                state["phase_start_time"] = now

            # 1. 右旋回（1.3秒）
            if state["phase"] == 1:
                if now - state["phase_start_time"] < 1.3:
                    print("Avoiding obstacle (right edge trace) - Phase 1: Right turn...")
                    target_x = (self.x1 + self.x2) // 2
                    return target_x, (80, 50), Mode.NVIDIA_FOLLOW
                state["phase"] = 2
                state["phase_start_time"] = now

            # 2. チェーン終了でリセット
            if state["phase"] == 2:
                self._state["nvidia_avoid_obstacle"] = {"phase": 0, "phase_start_time": None}
                print("Avoiding obstacle (right edge trace) - Complete, returning to normal mode...")
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.NVIDIA_FOLLOW
        elif yellow_pixel_count > 3000 and yellow_cx is not None and (right_pos is None or abs(right_pos) <= 5000):
            target_x = yellow_cx[0]  # X座標のみを取得
        elif right_x is not None:
            target_x = right_x
        else:
            target_x = (self.x1 + self.x2) // 2
        
        return target_x, None, Mode.NVIDIA_FOLLOW