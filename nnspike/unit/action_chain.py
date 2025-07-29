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
        # ROI_CNNをインスタンス変数に展開
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN

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
                return None, (40, 70), Mode.AVOID_OBSTACLE
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 右旋回（1.3秒）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 1.3:
                return None, (80, 50), Mode.AVOID_OBSTACLE
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
        0. 赤ボトル中心追従（red_pixel_countが一度3000以上となった時刻をred_detected_timeとし、red_detected_timeから3.0秒後に次段階へ遷移）
        1. 直進（2.0秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        2. 左旋回（1.5秒, 左:0, 右:30）
        3. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x, preference='left'）
        4. 直進（3.5秒, 両輪BASE_SPEED）
        5. 左旋回（1.5秒, 左:0, 右:30）
        6. 青検出（blue_pixel_countが一度1000以上→500以下になってから1.0秒後、または最大5秒でBACK_AND_TURN1に遷移）
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

        # 1. 赤ボトル中心追従（3.0秒、赤が見えなければ単に中央に進む）
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
            else:
                state["phase"] = 2
                state["phase_start_time"] = now

        # 2. 直進（2.0秒）
        if state["phase"] == 2:
            if now - state["phase_start_time"] < 2.0:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 3
                state["phase_start_time"] = now

        # 3. 左旋回（1.5秒, 左:0, 右:30）
        if state["phase"] == 3:
            if now - state["phase_start_time"] < 1.5:
                return None, (0, 30), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 4
                state["phase_start_time"] = now
                # フェーズ4突入時にpre_target_xを中央にリセット
                state["pre_target_x"] = (self.x1 + self.x2) // 2

        # 4. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x, preference='left'）
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
            else:
                state["phase"] = 5
                state["phase_start_time"] = now

        # 5. 直進（4.8秒）
        if state["phase"] == 5:
            if now - state["phase_start_time"] < 4.8:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            else:
                state["phase"] = 6
                state["phase_start_time"] = now

        # 6. 左旋回（1.5秒, 左:0, 右:30）
        if state["phase"] == 6:
            if now - state["phase_start_time"] < 1.5:
                return None, (0, 30), Mode.CARRY_BOTTLE1
            else:
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

        # 9. 青500以下になってから1秒間center追従、その後BACK_AND_TURN1
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
            if now - state["phase_start_time"] < 1.0:
                return target_x, None, Mode.CARRY_BOTTLE1
            else:
                state["pre_target_x"] = None
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
                return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN1
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 左旋回（3.0秒, 左:0, 右:30）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 3.0:
                return None, (0, 30), Mode.BACK_AND_TURN1
            else:
                state["phase"] = 2
                state["phase_start_time"] = now

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn1"] = {"phase": 0, "phase_start_time": None}
            return None, None, Mode.CARRY_BOTTLE2

        # どの分岐にも入らなかった場合のフェールセーフ
        return None, None, Mode.BACK_AND_TURN1

    def carry_bottle2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        0. 青ボトル中心追従（blue_pixel_countが一度3000以上になったらphase1へ）
        1. 3000以上の間はcenter追従。3000以下になったらphase2へ。
        2. 3000以下になってから0.5秒間center追従。その後phase3（左旋回2.8秒, speed=30）
        3. 直進（2.5秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        4. 左旋回（0.8秒, 左:0, 右:60）
        5. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x, preference='right'）
        6. 直進（2.0秒, 両輪BASE_SPEED）
        7. 左旋回（0.8秒, 左:0, 右:60）
        8. 青検出（1000超え→500以下で1秒後、または最大3秒でBACK_AND_TURN2）
        """
        state = self._state.setdefault("carry_bottle2", {
            "phase": 0,
            "phase_start_time": None,
            "blue_detected": False,
            "blue_lost_time": None,
            "pre_target_x": None,
            "blue_detected_time": None,
            "eye_blue_start": None,
            "blue_over1000": False,
            "blue_under500_time": None,
        })
        now = time.time()
        # 0. 右エッジトレースで青ピクセル数が3000を超えたらphase1へ
        if state["phase"] == 0:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            # 右エッジトレース（右端追従）しながら、青ピクセル数が3000を超えたらphase1へ遷移
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            if blue_pixel_count > 3000:
                state["phase"] = 1
                state["phase_start_time"] = now
                # phase遷移時はreturnしない（次のphase分岐で即座に動作）
            else:
                return target_x, None, Mode.CARRY_BOTTLE2

        # 1. 3000以上の間はcenter追従。3000以下になったらphase2へ
        if state["phase"] == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count > 3000 and center is not None:
                return center[0], None, Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 2
                state["phase_start_time"] = now
                state["blue_lost_time"] = now

        # 2. 3000以下になってから0.5秒間center追従。その後phase3（左旋回2.8秒, speed=30）
        if state["phase"] == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if now - state["phase_start_time"] < 0.5 and center is not None:
                return center[0], None, Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 3
                state["phase_start_time"] = now

        # 3. 左旋回（2.8秒, speed=30）
        if state["phase"] == 3:
            if now - state["phase_start_time"] < 2.8:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 4
                state["phase_start_time"] = now

        # 4. 直進（2.5秒, 両輪BASE_SPEED, pre_target_xも中央にリセット）
        if state["phase"] == 4:
            if now - state["phase_start_time"] < 2.5:
                state["pre_target_x"] = (self.x1 + self.x2) // 2
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 5
                state["phase_start_time"] = now

        # 5. 左旋回（1.5秒, 左:0, 右:30）
        if state["phase"] == 5:
            if now - state["phase_start_time"] < 1.5:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 6
                state["phase_start_time"] = now

        # 6. 仮想ライン直進（2.0秒, get_virtual_line_edges_at_y, previous_center_x=pre_target_x, preference='right'）
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
            else:
                state["phase"] = 7
                state["phase_start_time"] = now

        # 7. 直進（2.0秒, 両輪BASE_SPEED）
        if state["phase"] == 7:
            if now - state["phase_start_time"] < 2.0:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 8
                state["phase_start_time"] = now

        # 8. 左旋回（1.5秒, 左:0, 右:30）
        if state["phase"] == 8:
            if now - state["phase_start_time"] < 1.5:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 9
                state["phase_start_time"] = now

        # 9. 青検出（1000超えたらphase10へ）
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
            if blue_pixel_count > 1000:
                state["phase"] = 10
                state["phase_start_time"] = now
            return target_x, None, Mode.CARRY_BOTTLE2

        # 10. 青ピクセルが500以下まで減るまでcenter追従
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
            if blue_pixel_count <= 500:
                state["phase"] = 11
                state["phase_start_time"] = now
            return target_x, None, Mode.CARRY_BOTTLE2

        # 11. 500以下になってから1秒間center追従、その後back_and_turn2
        if state["phase"] == 11:
            blue_result = find_blue_target_center(image)
            if blue_result is not None:
                center, _, blue_pixel_count = blue_result
            else:
                center, blue_pixel_count = None, 0
            if now - state["phase_start_time"] < 1.0:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                state["pre_target_x"] = None
                state['eye_blue_start'] = None
                state['blue_over1000'] = False
                state['blue_under500_time'] = None
                state["phase"] = 0
                state["phase_start_time"] = None
                state["blue_detected"] = False
                return None, None, Mode.BACK_AND_TURN2

        # どの分岐にも入らなかった場合のフェールセーフ
        return None, None, Mode.CARRY_BOTTLE2

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
                return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN2
            else:
                state["phase"] = 1
                state["phase_start_time"] = now

        # 1. 1.5秒右旋回（左:30, 右:0）
        if state["phase"] == 1:
            if now - state["phase_start_time"] < 1.5:
                return None, (30, 0), Mode.BACK_AND_TURN2
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
            "reached": False,
            "reached_time": None,
            "turned": False,
            "blue_line_detected_time": None,
        })

        # 0. ライン到達前は中央追従
        if state["phase"] == 0:
            now = time.time()
            y_hit = get_line_trace_edges_at_x320(image)
            reached = y_hit is not None and y_hit >= 450
            if reached and not state["reached"]:
                state["reached"] = True
                state["reached_time"] = now
                state["phase"] = 1
                state["turned"] = False
            elif not state["reached"]:
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL
            # reachedが一度Trueになったら絶対にリセットしない
            # phase=1以降はreached状態を維持

        # 1. 到達直後0.5秒は直進
        if state["phase"] == 1:
            now = time.time()
            if state["reached_time"] is not None and (now - state["reached_time"] < 0.5):
                target_x = (self.x1 + self.x2) // 2
                return target_x, (BASE_SPEED, BASE_SPEED), Mode.HEAD_GOAL
            else:
                # 0.5秒経過後は必ず左旋回に遷移
                state["phase"] = 2
                state["phase_start_time"] = now
                state["left_turn_start"] = now
                return None, None, Mode.HEAD_GOAL

        # 2. 左旋回（1.5秒）
        if state["phase"] == 2:
            now = time.time()
            if "left_turn_start" not in state or state["left_turn_start"] is None:
                state["left_turn_start"] = now
            elapsed = now - state["left_turn_start"]
            if elapsed < 1.5:
                return None, (0, 30), Mode.HEAD_GOAL
            else:
                state["turned"] = True
                state["left_turn_start"] = None
                # 次のphaseへ
                state["phase"] = 3
                state["phase_start_time"] = now
                state["blue_line_detected_time"] = None

        # 3. 右エッジトレース。青ライン検出でフェーズ4へ遷移
        if state["phase"] == 3:
            now = time.time()
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (self.x1 + self.x2) // 2
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                state["phase"] = 4
                state["phase_start_time"] = now
                state["right_trace_after_blue_start"] = now
                return target_x, None, Mode.HEAD_GOAL
            return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、1.5秒右エッジトレースしたらPAUSE
        if state["phase"] == 4:
            now = time.time()
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if right_x is not None:
                target_x = right_x
            else:
                target_x = (self.x1 + self.x2) // 2
            if "right_trace_after_blue_start" not in state or state["right_trace_after_blue_start"] is None:
                state["right_trace_after_blue_start"] = now
            elapsed = now - state["right_trace_after_blue_start"]
            if elapsed >= 1.5:
                state["phase"] = 0
                state["phase_start_time"] = None
                return None, None, Mode.PAUSE
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
        state = self._state.setdefault("turn_at_end", {"reached": False, "turned": False, "reached_time": None})
        # get_line_trace_edges_at_x320でラインがoffset_y_for_turnに到達しているか判定
        y_hit = get_line_trace_edges_at_x320(frame)
        reached = y_hit is not None and y_hit >= offset_y_for_turn
        left_speed = right_speed = None
        target_x = None
        now = time.time()
        if not state["reached"] and reached:
            state["reached"] = True
            state["reached_time"] = now
            state["turned"] = False
        if not state["reached"]:
            # ライン到達前は中央
            target_x = (self.x1 + self.x2) // 2
            return target_x, (left_speed, right_speed), None
        # reachedになってから0.5秒間は直進
        elif state["reached"] and state["reached_time"] is not None and (now - state["reached_time"] < 0.5):
            target_x = (self.x1 + self.x2) // 2
            left_speed = right_speed = BASE_SPEED
            return target_x, (left_speed, right_speed), None
        # 0.5秒経過後に一度だけ左旋回
        elif not state["turned"]:
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
        （処理はコメントアウト中）
        """
        # state = self._state.setdefault("trun_left_gyro", {"start_accel_x": None})
        # gyro = self.et.last_spike_status.sensors.gyro
        # if not gyro or gyro.z is None:
        #     state["start_gyro_z"] = None
        #     return None, None, Mode.PAUSE
        # current_gyro_z = gyro.z
        # if state.get("start_gyro_z") is None:
        #     state["start_gyro_z"] = current_gyro_z
        # delta = current_gyro_z - state["start_gyro_z"]
        # if delta > -90:
        #     left_speed, right_speed = 0, 60
        #     return None, (left_speed, right_speed), Mode.TURN_LEFT_GYRO
        # state["start_gyro_z"] = None
        # return None, None, Mode.PAUSE
        pass

    def trun_right_gyro(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        ジャイロz角度の累積変化量（積分値）が+90度に達したらPAUSEに遷移する右旋回アクション。
        （処理はコメントアウト中）
        """
        # state = self._state.setdefault("trun_right_gyro", {"start_accel_x": None})
        # gyro = self.et.last_spike_status.sensors.gyro
        # if not gyro or gyro.z is None:
        #     state["start_gyro_z"] = None
        #     return None, None, Mode.PAUSE
        # current_gyro_z = gyro.z
        # if state.get("start_gyro_z") is None:
        #     state["start_gyro_z"] = current_gyro_z
        # delta = current_gyro_z - state["start_gyro_z"]
        # if delta < 90:
        #     left_speed, right_speed = 60, 0
        #     return None, (left_speed, right_speed), Mode.TURN_RIGHT_GYRO
        # state["start_gyro_z"] = None
        # return None, None, Mode.PAUSE
        pass