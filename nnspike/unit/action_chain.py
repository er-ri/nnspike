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
        backup/20250724/actions.pyと同じ障害物回避アクションチェーンを実行する。
        以下の順で動作する:
        1. 左旋回 (左:40, 右:70, 0.8秒)
        2. 右旋回 (左:80, 右:50, 1.3秒)
        3. コースに応じて左端または右端追従モードへ復帰
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

    def carry_bottle1(self, image: np.ndarray, color_reflected=None, color_ambient=None) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        carry_bottle1の動作シーケンス:
        1. FOLLOW_RIGHT_EDGE + find_bottle_center(red)
           - red_pixel_countが一度3000以上となった時刻をred_detected_timeとする
           - red_detected_timeから5.5秒後に次段階へ遷移
        2. TURN_LEFT（red_detected_time+5.5〜6.3秒, 0.8秒左旋回）
        3. GATE_PASS（red_detected_time+6.3〜9.3秒, 3.0秒間get_virtual_line_edges_at_yで直進）
        4. FORWARD（red_detected_time+9.3〜13.8秒, 4.5秒直進）
        5. TURN_LEFT（red_detected_time+13.8〜14.6秒, 0.8秒左旋回）
        6. EYE_BLUE（red_detected_time+14.6〜15.6秒, カラーセンサー値が指定範囲内になったら即停止。範囲外なら継続）
        """
        # 状態管理dictを利用
        state = self._state.setdefault("carry_bottle1", {"red_detected_time": None, "pre_target_x": None, "blue_lost_time": None})
        self.start_time = time.time() if self.start_time is None else self.start_time
        self.current_time = time.time()
        elapsed_time = self.current_time - self.start_time

        # base_powerを定義（必要に応じて調整可能）
        left_speed = right_speed = BASE_SPEED

        # color_valueを関数先頭で初期化（スコープエラー対策）
        color_value = None


        # 1. FOLLOW_RIGHT_EDGE + find_bottle_center(red)
        center, _, red_pixel_count = find_bottle_center(image=image, color="red")
        if state["red_detected_time"] is None:
            if red_pixel_count > 3000:
                state["red_detected_time"] = self.current_time
        # red_pixel_countが一度3000以上になってから5.5秒経過で次段階へ
        if state["red_detected_time"] is None or (self.current_time - state["red_detected_time"] < 5.5):
            if red_pixel_count > 3000 and center is not None:
                target_x = center[0]
            else:
                left_x, _, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
                x1, _, x2, _ = ROI_CNN
                target_x = left_x if left_x is not None else (x1 + x2) // 2
            left_speed = right_speed = BASE_SPEED
            return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 2. TURN_LEFT (red_detected_time+5.5〜6.3秒, 0.8秒左旋回)
        elif self.current_time - state["red_detected_time"] < 6.3:
            left_speed, right_speed = 0, 60
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 3. GATE_PASS (red_detected_time+6.3〜9.3秒, 3.0秒直進)
        elif self.current_time - state["red_detected_time"] < 9.3:
            pre_target_x = state.get("pre_target_x")
            temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x, preference='right')
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
            return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE1


        # 4. FORWARD (red_detected_time+9.3〜13.8秒, 4.5秒直進)
        elif self.current_time - state["red_detected_time"] < 13.8:
            left_speed = right_speed = BASE_SPEED
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1


        # 5. TURN_LEFT (red_detected_time+13.8〜14.6秒, 0.8秒左旋回)
        elif self.current_time - state["red_detected_time"] < 14.6:
            left_speed, right_speed = 0, 60
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 6. EYE_BLUE: カラーセンサー値で即停止（時間条件なし）
        # color_valueをここで取得
        if hasattr(self.et, 'color_value'):
            if callable(self.et.color_value):
                color_value = self.et.color_value()
            else:
                color_value = self.et.color_value
        center, _, blue_pixel_count = find_blue_target_center(image)
        x1, _, x2, _ = ROI_CNN
        color_value = color_value  # ダミー代入でスコープ明示
        if color_value is not None and 400 <= color_value <= 600:
            state["pre_target_x"] = None
            return None, None, Mode.PAUSE
        # 青ロストによる1秒待ち停止処理は削除（color_value判定のみで即停止）
        if center is not None:
            target_x = center[0]
        else:
            target_x = (x1 + x2) // 2
        left_speed = right_speed = BASE_SPEED
        return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 以降は停止または次のモードへ
        state["pre_target_x"] = None
        return None, None, Mode.CARRY_BOTTLE1

    def back_and_turn1(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        1. 2秒間後退（両輪BASE_SPEED）
        2. 1.6秒左旋回（左:0, 右:BASE_SPEED）
        3. 終了後PAUSE
        状態管理dictを利用。
        """
        state = self._state.setdefault("back_and_turn1", {"start_time": None, "phase": 0})
        now = time.time()
        if state["start_time"] is None:
            state["start_time"] = now
            state["phase"] = 0
        elapsed = now - state["start_time"]

        if state["phase"] == 0:
            # 2秒間後退
            if elapsed < 2.0:
                left_speed = right_speed = BASE_SPEED
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN1
            else:
                state["phase"] = 1
                state["start_time"] = now
                elapsed = 0.0
        if state["phase"] == 1:
            # 1.6秒左旋回
            if elapsed < 1.6:
                left_speed, right_speed = 0, BASE_SPEED
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN1
            else:
                # 終了: 状態リセット
                self._state["back_and_turn1"] = {"start_time": None, "phase": 0}
                return None, None, Mode.PAUSE

    def carry_bottle2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        carry_bottle2の動作シーケンス:
        1. FOLLOW_RIGHT_EDGE + find_bottle_center(blue)
           - blue_pixel_countが一度3000以上となった時刻をblue_detected_timeとする
           - blue_detected_timeから2.0秒後に次段階へ遷移
        2. TURN_RIGHT（blue_detected_time+2.0〜3.6秒, 1.6秒右旋回）
        3. FORWARD（blue_detected_time+3.6〜6.6秒, 3.0秒直進）
        4. TURN_LEFT（blue_detected_time+6.6〜7.4秒, 0.8秒左旋回）
        5. GATE_PASS（blue_detected_time+7.4〜11.4秒, 4.0秒get_virtual_line_edges_at_yで直進）
        6. TURN_LEFT（blue_detected_time+11.4〜12.2秒, 0.8秒左旋回）
        7. EYE_BLUE（blue_detected_time+12.2〜13.2秒, 青が消えてから1秒で停止。青を一度も検知していない場合はロスト判定しない）
        """
        state = self._state.setdefault("carry_bottle2", {"blue_detected_time": None, "pre_target_x": None, "blue_lost_time": None})
        self.start_time = time.time() if self.start_time is None else self.start_time
        self.current_time = time.time()
        elapsed_time = self.current_time - self.start_time

        left_speed = right_speed = BASE_SPEED


        # 1. FOLLOW_RIGHT_EDGE + find_bottle_center(blue)
        center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
        if state["blue_detected_time"] is None:
            if blue_pixel_count > 3000:
                state["blue_detected_time"] = self.current_time
        # blue_pixel_countが一度3000以上になってから2.0秒経過で次段階へ
        if state["blue_detected_time"] is None or (self.current_time - state["blue_detected_time"] < 2.0):
            if blue_pixel_count > 3000 and center is not None:
                target_x = center[0]
            else:
                left_x, _, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
                x1, _, x2, _ = ROI_CNN
                target_x = left_x if left_x is not None else (x1 + x2) // 2
            left_speed = right_speed = BASE_SPEED
            return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 2. TURN_RIGHT (blue_detected_time+2.0〜3.6秒, 1.6秒右旋回)
        elif self.current_time - state["blue_detected_time"] < 3.6:
            left_speed, right_speed = 60, 0  # 右旋回
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 3. FORWARD (blue_detected_time+3.6〜6.6秒, 3.0秒直進)
        elif self.current_time - state["blue_detected_time"] < 6.6:
            left_speed = right_speed = BASE_SPEED
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 4. TURN_LEFT (blue_detected_time+6.6〜7.4秒, 0.8秒左旋回)
        elif self.current_time - state["blue_detected_time"] < 7.4:
            left_speed, right_speed = 0, 60  # 左旋回
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE2


        # 3. GATE_PASS (red_detected_time+5.8〜8.8秒, 3.0秒直進)
        elif self.current_time - state["red_detected_time"] < 8.8:
            pre_target_x = state.get("pre_target_x")
            temp_x = get_virtual_line_edges_at_y(image, OFFSET_Y, previous_center_x=pre_target_x, preference='right')
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
            return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 4. FORWARD (red_detected_time+8.8〜13.3秒, 4.5秒直進)
        elif self.current_time - state["red_detected_time"] < 13.3:
            left_speed = right_speed = BASE_SPEED
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 5. TURN_LEFT (red_detected_time+13.3〜14.1秒, 0.8秒左旋回)
        elif self.current_time - state["red_detected_time"] < 14.1:
            left_speed, right_speed = 0, 60
            return None, (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # 6. EYE_BLUE (red_detected_time+14.1〜15.1秒, カラーセンサー値で即停止)
        elif self.current_time - state["red_detected_time"] < 15.1:
            # カラーセンサー値で即停止判定
            # color_valueのみで判定、範囲は400〜600（幅広・霧の良い数字）
            if color_value is not None and 400 <= color_value <= 600:
                state["pre_target_x"] = None
                return None, None, Mode.PAUSE
            # それ以外は従来通り青追従
            center, _, blue_pixel_count = find_blue_target_center(image)
            x1, _, x2, _ = ROI_CNN
            if center is not None:
                target_x = center[0]
            else:
                target_x = (x1 + x2) // 2
            left_speed = right_speed = BASE_SPEED
            return target_x, (left_speed, right_speed), Mode.CARRY_BOTTLE1
    def back_and_turn2(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        以下の順で動作する:
        1. 2秒間後退（両輪BASE_SPEED）
        2. 0.8秒右旋回（左:BASE_SPEED, 右:0）
        3. 終了後PAUSE
        状態管理dictを利用。
        """
        state = self._state.setdefault("back_and_turn2", {"start_time": None, "phase": 0})
        now = time.time()
        if state["start_time"] is None:
            state["start_time"] = now
            state["phase"] = 0
        elapsed = now - state["start_time"]

        if state["phase"] == 0:
            # 2秒間後退
            if elapsed < 2.0:
                left_speed = right_speed = BASE_SPEED
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN2
            else:
                state["phase"] = 1
                state["start_time"] = now
                elapsed = 0.0
        if state["phase"] == 1:
            # 0.8秒右旋回
            if elapsed < 0.8:
                left_speed, right_speed = BASE_SPEED, 0
                return None, (left_speed, right_speed), Mode.BACK_AND_TURN2
            else:
                # 終了: 状態リセット
                self._state["back_and_turn2"] = {"start_time": None, "phase": 0}
                return None, None, Mode.PAUSE

    def heading_goal(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        ゴールに向かう際、turn_at_endと同じ動作を順次実行する。
        以下の順で動作する:
        1. ライン到達前は中央追従
        2. 到達時に一度だけ左旋回（0, 60, 0.8秒）
        3. 以降は右端追従（ライン消失時はPAUSE）
        """
        state = self._state.setdefault("heading_goal", {"reached": False, "turned": False})
        x1, y1, x2, y2 = ROI_CNN
        y_hit = get_line_trace_edges_at_x320(image)
        reached = y_hit is not None and y_hit >= 400
        left_speed = right_speed = None
        target_x = None
        if not state["reached"] and reached:
            state["reached"] = True
            state["turned"] = False
        if not state["reached"]:
            # ライン到達前は中央
            target_x = (x1 + x2) // 2
            return target_x, (left_speed, right_speed), Mode.HEAD_GOAL
        elif not state["turned"]:
            # 到達した瞬間に一度だけ左旋回
            elapsed_time = getattr(self, '_heading_goal_turn_start', None)
            if elapsed_time is None:
                self._heading_goal_turn_start = time.time()
                elapsed_time = self._heading_goal_turn_start
            if time.time() - elapsed_time < 0.8:
                left_speed, right_speed = 0, 60
                return target_x, (left_speed, right_speed), Mode.HEAD_GOAL
            else:
                del self._heading_goal_turn_start
                state["turned"] = True
                return target_x, (0, 0), Mode.HEAD_GOAL
        else:
            # 右端追従の動作をここで実装し、ライン完全消失時にPAUSEで停止する
            x1, _, x2, _ = ROI_CNN
            # mask: ライン検出用の2値化画像（ライン部分が白=255, それ以外は黒=0 のnumpy配列）
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if left_x is None and right_x is None and np.count_nonzero(mask) == 0:
                # ライン完全消失で停止
                self._state["heading_goal"] = {"reached": False, "turned": False, "paused": False}
                return None, None, Mode.PAUSE
            if not state.get("paused", False):
                # 右端追従
                if right_x is not None:
                    target_x = right_x
                else:
                    target_x = (x1 + x2) // 2
                left_speed = right_speed = BASE_SPEED
                state["paused"] = True
                return target_x, (left_speed, right_speed), Mode.HEAD_GOAL
            else:
                # 2回目以降は完全停止
                self._state["heading_goal"] = {"reached": False, "turned": False, "paused": False}
                return None, None, Mode.PAUSE

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