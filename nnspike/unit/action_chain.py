import time  # 時間計測用
from typing import Optional, Tuple  # 型ヒント用

import numpy as np  # 画像処理用

# 定数・モード・ROI設定
from nnspike.constants import OFFSET_Y, ROI_CNN, Mode, BASE_SPEED, ROI_LINE_HORIZON3, ROI_LOOP, ROI_LINE_CORNER

# --- 閾値定数（全体で統一管理） ---
BLUE_AREA_MAX_THRESHOLD = 18000
BLUE_AREA_MIN_THRESHOLD = 3000
LEFT_POS_THRESHOLD_PHASE0 = 13000
LEFT_POS_THRESHOLD_PHASE2 = 17000
LEFT_POS_THRESHOLD_PHASE4 = 19000
LEFT_POS_THRESHOLD_PHASE6 = 21000
LEFT_POS_THRESHOLD_PHASE8 = 22000

# ロボット本体クラス
from nnspike.unit.etrobot import ETRobot
# 画像処理・ライン/ターゲット検出関数群
from nnspike.utils.control import (
    find_bottle_center,  # ボトル中心座標・ピクセル数検出
    get_line_edges_at_y,  # 指定Y座標でのライン左右端検出
    get_virtual_line_target_x,  # 仮想ライン左右端検出
    find_blue_target_center,  # 青ターゲット中心座標・ピクセル数検出
    get_is_blue_line_at_y,  # 指定Y座標での青ライン有無判定
    is_x320_on_blue_target,  # 画像中央x=320付近で青ターゲット検出
    is_x320_on_red_target,  # 画像中央x=320付近で赤ターゲット検出
    get_red_target_center_x,  # 赤ターゲット中心x座標取得
    is_left_black_line_detected,  # 左黒ライン検出
    is_lower_horizontal_line_detected,  # 下部水平黒ライン検出
    is_vertical_black_line_detected,  # 垂直黒ライン検出
    is_upper_horizontal_line_detected,  # 上部水平黒ライン検出
    get_blue_line_pixel,  # 青オブジェクト面積検出
    is_fast_corner_detected,  # コーナー検出
)

# 型ヒント用: 3要素タプル明示
SpeedTuple = Tuple[int, int, int]

class PhaseManager:
    """フェーズ管理クラス。"""

    def __init__(self):
        """PhaseManagerの初期化処理."""
        self._state = {}
        self._state["phase"] = 0
        self._state["position_start"] = None

    def get_phase(self) -> int:
        """現在のphase値を取得する."""
        phase = self._state.get("phase", 0)
        return phase

    def next_phase(self, skip: int = 1) -> None:
        """phase値をskip分進める（デフォルト1）。"""
        self._state["phase"] = self._state.get("phase", 0) + skip

    def set_position_start(self, key: str, value) -> None:
        """指定したkey（例: 'right_position_start'）にvalue（例: モーター位置）をセットする。

        valueがint型以外の場合は0に変換してセットする。
        """
        # valueがtuple型の場合は先頭要素をintとして扱う
        if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], int):
            value = value[0]
        if not isinstance(value, int):
            value = 0
        self._state[key] = value

    def get_position_start(self, key: str) -> int:
        """指定したkeyのposition_start値を取得する（必ずint型で返す）."""
        value = self._state.get(key, None)
        if isinstance(value, int):
            return value
        if isinstance(value, tuple):
            if len(value) == 0:
                return 0
            if isinstance(value[0], int):
                return value[0]
            return 0
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

class ActionChain(object):
    """ETRobotのためのアクションシーケンス管理クラス."""

    def __init__(self, et: ETRobot, course: str, course_type: str) -> None:
        """ActionChainの初期化処理."""
        self.et = et  # ロボット本体
        self.course = course  # コース種別
        # コース種別の逆コースを定義
        if course == "right":
            self.opposite_course = "left"
        else:
            self.opposite_course = "right"
        self.course_type = course_type  # 上段/下段コース（デフォルトupper）
        self.start_time = 0.0  # アクション開始時刻
        self.current_time = 0.0  # 現在時刻
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN  # ROI座標
        self._init = False
        self.pre_target_x = (self.x1 + self.x2) // 2

    def initialize_action(self, motor_side: str = "right"):
        """アクション開始時の状態初期化処理.

        motor_side: "right"または"left"で初期位置記録対象を指定する。
        """
        self._phase = PhaseManager()
        self._status = self.get_motor_position(mode="status")
        self._phase.set_position_start("position_start", self.get_motor_position(motor_side, status=self._status))
        self._init = True

    def reset_action(self):
        """アクション終了時の状態リセット処理."""
        self._init = False
        self._status = None

    def get_motor_position(self, motor_side: str = "right", mode: str = "position", status=None) -> int:
        """モーター位置・status取得メソッド.

        motor_side='right'で右モータ(B)、'left'で左モータ(A)のrelative_positionを返す。
        mode='position'なら該当モータのrelative_position（絶対値, Noneなら0）、'status'ならstatusオブジェクト。
        status引数を指定すればそれを使い、未指定時のみ内部で取得する。
        負荷軽減のため、複数回呼び出し時はstatusを外部で取得・使い回すこと。
        ただしmode='status'時は必ず最新statusを再取得する。
        """
        # statusがint型の場合はNoneに置き換え
        if isinstance(status, int):
            status = None
        if mode == "status":
            # 必ず最新statusを取得
            status = self.et.get_spike_status()
            if status is None:
                print("[get_motor_position] get_spike_status() returned None")
                return 0
            # status型はint型ではないので、mode='status'時は0を返す
            return 0
        if status is None:
            status = self.et.get_spike_status()
            if status is None:
                print("[get_motor_position] get_spike_status() returned None")
                return 0
        if mode == "position":
            motor_key = "B" if motor_side == "right" else "A"
            if status is not None and status.motors.get(motor_key) is not None:
                pos = status.motors[motor_key].relative_position
                if isinstance(pos, int):
                    return abs(pos)
                elif isinstance(pos, tuple):
                    # すべての要素がint型の場合は先頭要素を返す
                    if len(pos) > 0 and isinstance(pos[0], int):
                        return abs(pos[0])
                    # tupleの中身がint型でない場合は0
                    return 0
                else:
                    print(f"[get_motor_position] {motor_key} position invalid: {pos} (return 0)")
                    return 0
            print(f"[get_motor_position] status or motor_key invalid (return 0)")
            return 0
        
        print("[get_motor_position] Unexpected state reached.")
        return 0

    def get_target_x_by_course(self, image, offset_y, course="right"):
        """
        image, offset_y, course("right"/"left")を受けてtarget_xを返す共通メソッド
        """
        if course == "right":
            _, right_x, _ = get_line_edges_at_y(image, ROI_CNN, offset_y, 80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
        elif course == "left":
            left_x, _, _ = get_line_edges_at_y(image, ROI_CNN, offset_y, 80)
            target_x = left_x if left_x is not None else (self.x1 + self.x2) // 2
        else:
            target_x = (self.x1 + self.x2) // 2
        return target_x

    def small_turn_left(self) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """短時間（0.3秒）左旋回アクション.

        0.3秒間、左モータ:0・右モータ:50で旋回し、0.3秒経過後にPAUSEへ遷移。
        戻り値: (None, (左速度, 右速度), モード)
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.3:
            return None, (0, 50, 0), Mode.SMALL_TURN_LEFT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def small_turn_right(self) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """短時間（0.3秒）右旋回アクション.

        0.3秒間、左モータ:50・右モータ:0で旋回し、0.3秒経過後にPAUSEへ遷移。
        戻り値: (None, (左速度, 右速度), モード)
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.3:
            return None, (50, 0, 0), Mode.SMALL_TURN_RIGHT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def blue_bottle_catch(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """青ボトルキャッチモード.

        ・phase0: 青ターゲット中心x座標へ追従（青ピクセル数1000超えたらphase1へ）
        ・phase1: 青ピクセル数1000以上の間は中心x座標へ追従、500以下でphase2へ（右モーター位置記録）
        ・phase2: 青ピクセル数500以下になってから右モーター300ユニット移動まで中心x座標へ追従、300到達でphase3へ
        ・phase3: 状態リセットしPAUSEへ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # phase0: 青ターゲット中心x座標へ追従（青ピクセル数1000超えたらphase1へ）
        if phase.get_phase() == 0:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                phase.next_phase()
            else:
                return target_x, None, Mode.BLUE_BOTTLE_CATCH

        # phase1: 青ピクセル数1000以上の間は中心x座標へ追従、500以下でphase2へ（右モーター位置記録）
        if phase.get_phase() == 1:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                phase.next_phase()
                # phase2用 右モーター相対位置記録
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.BLUE_BOTTLE_CATCH

        # phase2: 青ピクセル数500以下になってから右モーター300ユニット移動まで中心x座標へ追従、300到達でphase3へ
        if phase.get_phase() == 2:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 300:
                return target_x, None, Mode.BLUE_BOTTLE_CATCH
            else:
                phase.next_phase()
        # phase3: 状態リセットしPAUSEへ遷移
        if phase.get_phase() == 3:
            self.reset_action()
            return None, None, Mode.PAUSE

        print("[blue_bottle_catch] Unexpected state reached.")
        return None, None, Mode.PAUSE

    def turn_left_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side='right')
        phase = self._phase
        status = self._status
        """
        左旋回（右モーターBの相対位置差分で判定）。430未満の間は左:0,右:30で継続。430超えたらPAUSE。
        """
        right_position = self.get_motor_position('right', status=status)
        if abs(right_position - phase.get_position_start('position_start')) > 430:
            self.reset_action()
            return None, None, Mode.PAUSE
        return None, (0, 30, 0), Mode.TURN_LEFT_RELATIVE

    def turn_right_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side='left')
        phase = self._phase
        status = self._status
        """
        右旋回（左モーターAの相対位置差分で判定）。430未満の間は左:30,右:0で継続。430超えたらPAUSE。
        """
        left_position = self.get_motor_position('left', status=status)
        if abs(left_position - phase.get_position_start('position_start')) > 430:
            self.reset_action()
            return None, None, Mode.PAUSE
        return None, (30, 0, 0), Mode.TURN_RIGHT_RELATIVE

    def avoid_obstacle_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        障害物回避の相対位置判定バージョン。
        ・phase0: 左旋回（右モーター500未満まで、左:40,右:70/左:70,右:40）。到達でphase1へ、右モーター位置記録。
        ・phase1: intersection_y=450で黒水平ライン検出まで中央追従（最低500進める）。500未満は中央追従、500以上で黒ライン検出判定。検出でphase2へ、右モーター位置記録。
        ・phase2: 右モーター移動距離300未満なら中央追従、300以上でphase3へ、右モーター位置記録。
        ・phase3: 左旋回（250未満は(40,70)/(70,40)、250以上350未満は垂直黒ライン検出で即phase4へ、350以上は強制的にphase4へ）
        ・phase4: 状態リセットし右端/左端追従モード(FOLLOW_RIGHT_EDGE/FOLLOW_LEFT_EDGE)へ復帰
        戻り値: (None, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # phase0: 左旋回（右モーター500未満まで、左:40,右:70/左:70,右:40）。到達でphase1へ、右モーター位置記録。
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 500:
                if self.course == "right":
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase1: intersection_y=450で黒水平ライン検出まで中央追従（最低500進める）。500未満は中央追従、500以上で黒ライン検出判定。検出でphase2へ、右モーター位置記録。
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 400:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
            if is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3):
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE

        # phase2: 右モーター移動距離300未満なら中央追従、300以上でphase3へ、右モーター位置記録。
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 150:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase3: 左旋回（50未満は(30,60)/(60,30)、500未満は(30,60)/(60,30)、500以上で次フェーズへ。500未満かつ垂直黒ライン検出で次フェーズへ）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            distance = abs(current_pos - position_start)
            if distance < 50:
                if self.course == "right":
                    return None, (30, 60, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (60, 30, 0), Mode.AVOID_OBSTACLE
            elif distance < 500:
                if is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                    phase.next_phase()
                else:
                    if self.course == "right":
                        return None, (30, 60, 0), Mode.AVOID_OBSTACLE
                    else:
                        return None, (60, 30, 0), Mode.AVOID_OBSTACLE
            else:
                phase.next_phase()

        # phase4: 状態リセットし右端/左端追従モード(FOLLOW_RIGHT_EDGE/FOLLOW_LEFT_EDGE)へ復帰
        if phase.get_phase() == 4:
            self.reset_action()
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            if self.course == "right":
                return target_x, None, Mode.FOLLOW_RIGHT_EDGE
            else:
                return target_x, None, Mode.FOLLOW_LEFT_EDGE

        print("[avoid_obstacle_relative] Unexpected state reached.")
        return None, None, Mode.AVOID_OBSTACLE

    def high_speed_avoid(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:

        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # phase0: 黄色領域検出で次フェーズへ。未検出時は中央追従・HIGH_SPEED_AVOID返却
        if phase.get_phase() == 0:
            print(f"[DEBUG] phase=0 yellow_pixel_count={find_bottle_center(image=image, color='yellow')[2]}")
            _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
            if yellow_pixel_count > 5000:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, 100), Mode.HIGH_SPEED_AVOID

        # phase1: 黄色領域検出で次フェーズへ。未検出時は中心または中央追従・HIGH_SPEED_AVOID返却
        if phase.get_phase() == 1:
            yellow_cx, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
            print(f"[DEBUG] phase=1 yellow_pixel_count={yellow_pixel_count} yellow_cx={yellow_cx}")
            if yellow_pixel_count > 18000:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if yellow_cx is not None:
                    target_x = yellow_cx[0]
                else:
                    target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, BASE_SPEED), Mode.HIGH_SPEED_AVOID

        # phase2: 左旋回（右モーター500未満まで、左:40,右:70/左:70,右:40）。到達でphase3へ、右モーター位置記録。
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 500:
                if self.course == "right":
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase3: intersection_y=450で黒水平ライン検出まで中央追従（最低500進める）。500未満は中央追従、500以上で黒ライン検出判定。検出でphase4へ、右モーター位置記録。
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 400:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
            if is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3):
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID

        # phase4: 右モーター移動距離150未満なら中央追従、150以上でphase5へ、右モーター位置記録。
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 150:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase5: 左旋回（50未満は(30,60)/(60,30)、500未満は(30,60)/(60,30)、500以上で次フェーズへ。500未満かつ垂直黒ライン検出で次フェーズへ）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            distance = abs(current_pos - position_start)
            if distance < 50:
                if self.course == "right":
                    return None, (30, 60, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (60, 30, 0), Mode.HIGH_SPEED_AVOID
            elif distance < 500:
                if is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                    phase.next_phase()
                else:
                    if self.course == "right":
                        return None, (30, 60, 0), Mode.HIGH_SPEED_AVOID
                    else:
                        return None, (60, 30, 0), Mode.HIGH_SPEED_AVOID
            else:
                phase.next_phase()

        # phase6: コーナー検出（is_fast_corner_detected）で次フェーズへ。未検出時は中央追従・HIGH_SPEED_AVOID返却
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=6 position_diff={position_diff}")
            corner_detected = is_fast_corner_detected(image, course=self.course)
            print(f"[DEBUG] phase=6 コーナー検出: {corner_detected}")
            if corner_detected and position_diff >= 400:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, BASE_SPEED), Mode.HIGH_SPEED_AVOID

        # phase7: 右モーター移動距離・垂直黒ライン判定で旋回継続または次フェーズへ
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=7 position_diff={position_diff}")
            if position_diff < 700 and not is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                if self.course == "right":
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
            phase.next_phase()
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase8: 青面積判定で状態リセット・DOUBLE_LOOPまたはHIGH_SPEED_AVOID継続
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=8 position_diff={position_diff}")
            corner_detected = is_fast_corner_detected(image, course=self.course)
            print(f"[DEBUG] phase=8 コーナー検出: {corner_detected}")
            if corner_detected and position_diff >= 2500:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, 100), Mode.HIGH_SPEED_AVOID

        # phase9: 右モーター移動距離・垂直黒ライン判定で旋回継続または次フェーズへ
        if phase.get_phase() == 9:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=9 position_diff={position_diff}")
            if position_diff < 700 and not is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                if self.course == "right":
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
            phase.next_phase()
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase10: 青面積判定または右モーター500進んだらDOUBLE_LOOP、そうでなければHIGH_SPEED_AVOID継続
        if phase.get_phase() == 10:
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            blue_area = get_blue_line_pixel(image)
            print(f"[DEBUG] phase=10 blue_area={blue_area}")
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=10 position_diff={position_diff}")
            if blue_area > BLUE_AREA_MAX_THRESHOLD or position_diff >= 500:
                self.reset_action()
                return target_x, None, Mode.DOUBLE_LOOP
            else:
                return target_x, (0, 0, 100), Mode.HIGH_SPEED_AVOID

        print("[high_speed_avoid] Unexpected state reached.")
        return None, None, Mode.HIGH_SPEED_AVOID

    def carry_bottle1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        carry_bottle1の位置判定バージョン。
        実装内容に完全一致:
        0. 右エッジトレース（赤ピクセル数が3000を超えたらphase1へ、右モーター初期位置記録）
        1. 赤ボトル中心追従（右モーター相対位置差分が1000未満の間、赤ピクセル500未満なら中央。1000超えたらphase2へ、右モーター位置記録）
        2. 左エッジトレース（右モーター相対位置差分が1200未満の間、1200超えたらphase3へ、右モーター位置記録）
        3. 左旋回（右モーター相対位置差分が390未満の間 左:0, 右:30。390超えたらphase4へ、右モーター位置記録）
        4. 直進（右モーター100ユニット移動まで。100超えたらphase5へ、右モーター位置記録、pre_target_x初期化）
        5. 仮想ライン直進（右モーター1000ユニット移動まで、get_virtual_line_target_xで中心追従、pre_target_x更新。1000超えたらphase6へ、右モーター位置記録）
        6. 直進（右モーター1900ユニット移動まで。1900超えたらphase7へ、右モーター位置記録）
        7. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最低300・最大右モーター500ユニット。条件満たせばphase8へ、右モーター位置記録）
        8. 青検出（青ピクセル数1000超えたらphase9へ）
        9. 青1000以上の間center追従、500以下でphase10へ、右モーター位置記録
        10. 青500以下になってから右モーター300ユニット移動までcenter追従。300超えたらphase11へ
        11. 状態リセットしBACK_AND_TURN1へ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 右エッジトレース（赤ピクセル数が3000を超えたらphase1へ、右モーター初期位置記録）
        if phase.get_phase() == 0:
            #left_x, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold=80)
            #target_x = right_x if self.course == "right" else left_x
            #if target_x is None:
            #    target_x = (self.x1 + self.x2) // 2
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            _, _, red_pixel_count = find_bottle_center(image=image, color="red")
            if red_pixel_count > 3000:
                phase.next_phase()
                # phase1用 右モーター相対位置記録（絶対値）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 1. 赤ボトル中心追従（右モーター相対位置差分が1000未満の間、赤ピクセル500未満なら中央。1000超えたらphase2へ、右モーター位置記録）
        if phase.get_phase() == 1:
            center, _, red_px = find_bottle_center(image=image, color="red")
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            # 右モーター相対位置差分で継続判定
            if abs(current_pos - position_start) < 1000:
                if center is not None and red_px is not None and red_px >= 500:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE1
            # 1000超えたら次フェーズへ
            phase.next_phase()
            # phase2用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 2. 左エッジトレース（右モーター相対位置差分が1200未満の間、1200超えたらphase3へ、右モーター位置記録）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 1200 if self.course_type == "upper" else 700
            if abs(current_pos - position_start) < threshold:
                #left_x, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=300, threshold=80)
                #target_x = left_x if self.course == "right" else right_x
                #if target_x is None:
                #    target_x = (self.x1 + self.x2) // 2
                target_x = self.get_target_x_by_course(image, offset_y=300, course=self.opposite_course)
                return target_x, None, Mode.CARRY_BOTTLE1
            # 1200超えたら次フェーズへ
            phase.next_phase()
            # phase3用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 3. 左旋回（右モーター相対位置差分が390未満の間 courseに応じて左:0,右:30または左:30,右:0。390超えたらphase4へ、右モーター位置記録）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 390:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE1
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE1
            # 390超えたら次フェーズへ
            phase.next_phase()
            # phase4用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 4. 直進（右モーター200ユニット移動まで。200超えたらphase5へ、右モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 200:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE1
            # 300超えたら次フェーズへ
            phase.next_phase()
            # phase5用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            self.pre_target_x = (self.x1 + self.x2) // 2

        # 5. 仮想ライン直進（右モーター1500ユニット移動まで、get_virtual_line_target_xで中心追従、pre_target_x更新。1000超えたらphase6へ、右モーター位置記録）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 1500:
                # 右に障害物がある場合は左回避を明示
                temp_x = get_virtual_line_target_x(image, previous_center_x=self.pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    self.pre_target_x = temp_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    self.pre_target_x = target_x
                return target_x, None, Mode.CARRY_BOTTLE1
            # 1000超えたら次フェーズへ
            phase.next_phase()
            # phase6用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 6. 直進（右モーター1300ユニット移動まで。1300超えたらphase7へ、右モーター位置記録）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 1300:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE1
            # 1300超えたら次フェーズへ
            phase.next_phase()
            # phase7用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 7. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最低300・最大右モーター500ユニット。条件満たせばphase8へ、右モーター位置記録）
        if phase.get_phase() == 7:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_rotation_done = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_rotation_done = position_diff >= 300
            max_limit = 400 if self.course_type == "lower" else 500
            position_limit_reached = position_diff >= max_limit
            
            # 最低300ユニット旋回後にblue_target検出、または500ユニット到達で次へ
            if (minimum_rotation_done and blue_target_detected) or position_limit_reached:
                if self.course_type == "lower":
                    phase.next_phase(skip=2)  # スキップ
                else:
                    phase.next_phase()
            else:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE1
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE1

        # 8. 青検出（青ピクセル数1000超えたらphase9へ）
        if phase.get_phase() == 8:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                phase.next_phase()
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 9. 青1000以上の間center追従、500以下でphase10へ、右モーター位置記録
        if phase.get_phase() == 9:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                phase.next_phase()
                # phase10用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 10. 青が一定値以下になってから右モーターが所定の移動量に達するまでcenter追従。条件を満たしたら次のphaseへ
        if phase.get_phase() == 10:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2

            # 右モーター位置差分で継続判定（upper:300, lower:200）
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 300 if self.course_type == "upper" else 200
            if abs(current_pos - position_start) < threshold:
                return target_x, None, Mode.CARRY_BOTTLE1
            else:
                phase.next_phase()
            
        # 11. 状態リセットしBACK_AND_TURN1へ遷移
        if phase.get_phase() == 11:
            self.reset_action()
            return None, None, Mode.BACK_AND_TURN1

        print("[carry_bottle1_relative] Unexpected state reached.")
        return None, None, Mode.CARRY_BOTTLE1

    def back_and_turn1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        back_and_turn1の位置判定バージョン。
        実装内容に完全一致:
        0. 後退（右モーター600ユニット移動まで。600超えたらphase1へ、右モーター位置記録）
        1. 左旋回（最低右モーター450ユニットは必ず旋回。450ユニット超えてからis_x320_on_red_target(image, x_tolerance=60)検出または940ユニット到達まで左:0,右:30で継続。条件満たせばphase2へ）
        2. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        戻り値: (None, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 後退（右モーター600ユニット移動まで。600超えたらphase1へ、右モーター位置記録）
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 600:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.BACK_AND_TURN1
            phase.next_phase()
            # phase1用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 1. 左旋回（最低右モーター450ユニットは必ず旋回。450ユニット超えてからis_x320_on_red_target(image, x_tolerance=60)検出または940ユニット到達まで左:0,右:30で継続。条件満たせばphase2へ）
        if phase.get_phase() == 1:
            red_target_detected = is_x320_on_red_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_position_reached = False
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 450
            position_limit_reached = position_diff >= 940
        
            # 最低450ユニットは必ず旋回
            if not minimum_position_reached:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.BACK_AND_TURN1
                else:
                    return None, (30, 0, 0), Mode.BACK_AND_TURN1
            # 450ユニット超えてから、ターゲット検出または940ユニット到達まで継続
            if (not red_target_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.BACK_AND_TURN1
                else:
                    return None, (30, 0, 0), Mode.BACK_AND_TURN1
            phase.next_phase()

        # 2. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        if phase.get_phase() == 2:
            self.reset_action()
            return None, None, Mode.CARRY_BOTTLE2

        print("[back_and_turn1_relative] Unexpected state reached.")
        return None, None, Mode.BACK_AND_TURN1

    def carry_bottle2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        carry_bottle2の位置判定バージョン。
        実装内容に完全一致:
        0. 赤ターゲット中心追従（青ピクセル数20000未満の間は赤中心追従、20000以上でphase1へ）
        1. 青ボトル中心追従（青ピクセル数2000以上の間center追従、2000未満でphase2へ、右モーター位置記録）
        2. 右モーター200ユニット移動までcenter追従。200超えたらphase3へ、右モーター位置記録
        3. 左黒ライン検出まで左旋回。最大右モーター1000ユニット。検出または1000超えたらphase4へ、右モーター位置記録
        4. 直進（右モーター870ユニット移動まで、両輪BASE_SPEED。870超えたらphase5へ、右モーター位置記録）
        5. 左旋回（右モーター380ユニット移動まで、左:0,右:30。380超えたらphase6へ、右モーター位置記録）
        6. 直進（右モーター200ユニット移動まで、両輪BASE_SPEED。200超えたらphase7へ、右モーター位置記録、pre_target_x初期化）
        7. 仮想ライン直進（右モーター800ユニット移動まで、get_virtual_line_target_xで中心追従、pre_target_x更新。800超えたらphase8へ、右モーター位置記録）
        8. 直進（右モーター900ユニット移動まで、両輪BASE_SPEED。900超えたらphase9へ、右モーター位置記録）
        9. 左旋回（青ターゲット検出まで、最低右モーター300、最大500ユニット、左:0,右:30。条件満たせばphase10へ、右モーター位置記録）
        10. 青検出（青ピクセル数1000超えたらphase11へ、最大右モーター400ユニット。条件満たせば右モーター位置記録）
        11. 青ピクセルが500以下まで減るまでcenter追従（500以下でphase12へ、最大右モーター400ユニット。条件満たせば右モーター位置記録）
        12. 右モーター300ユニット移動までcenter追従。300超えたらphase13へ
        13. 状態リセットしBACK_AND_TURN2へ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 赤ターゲット中心追従（青ピクセル数20000未満の間は赤中心追従、20000以上でphase1へ）
        if phase.get_phase() == 0:
            _, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count < 18000:
                # 赤ターゲット中心追従
                red_center_x = get_red_target_center_x(image)
                target_x = red_center_x if red_center_x is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                phase.next_phase()

        # 1. 青ボトル中心追従（青ピクセル数2000以上の間center追従、2000未満でphase2へ、右モーター位置記録）
        if phase.get_phase() == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
            if blue_pixel_count >= 4000:
                # 青ボトル中心x座標へ追従
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                # 青ピクセル数が2000未満になった瞬間phase2へ
                phase.next_phase()
                # phase2用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 2. 右モーター200ユニット移動までcenter追従。200超えたらphase3へ、右モーター位置記録
        if phase.get_phase() == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 200:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase3用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 3. 左黒ライン検出まで左旋回。最大右モーター1000ユニット。検出または1000超えたらphase4へ、右モーター位置記録
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            min_limit = 700
            max_limit = 1500
            position_delta = abs(current_pos - position_start)
            position_limit_reached = position_delta >= max_limit

            # 500未満は検知開始しない
            if position_delta < min_limit and not position_limit_reached:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2

            # 500以上になったら判定開始
            line_detected = is_left_black_line_detected(image, self.course)

            if (not line_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase4用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 4. 直進（右モーター870ユニット移動まで、両輪BASE_SPEED。870超えたらphase5へ、右モーター位置記録）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 870 if self.course_type == "upper" else 1300
            if abs(current_pos - position_start) < threshold:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase5用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 5. 左旋回（右モーター380ユニット移動まで courseに応じて左:0,右:30または左:30,右:0。380超えたらphase6へ、右モーター位置記録）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 350:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase6用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 6. 直進（右モーター200ユニット移動まで、両輪BASE_SPEED。200超えたらphase7へ、右モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 100:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase7用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            self.pre_target_x = (self.x1 + self.x2) // 2

        # 7. 仮想ライン直進（右モーター1300ユニット移動まで、get_virtual_line_target_xで中心追従、pre_target_x更新。800超えたらphase8へ、右モーター位置記録）
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 1400:
                temp_x = get_virtual_line_target_x(image, previous_center_x=self.pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    self.pre_target_x = temp_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    self.pre_target_x = target_x
                return target_x, None, Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase8用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 8. 直進（右モーター400ユニット移動まで、両輪BASE_SPEED。900超えたらphase9へ、右モーター位置記録）
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 400:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase9用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 9. 左旋回（青ターゲット検出まで、最低右モーター300、最大500ユニット、左:0,右:30または左:30,右:0。条件満たせばphase10へ、右モーター位置記録）
        if phase.get_phase() == 9:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_position_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 300
            position_limit_reached = position_diff >= 500
            # 最低300ユニットは必ず旋回
            if not minimum_position_reached:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            # 300ユニット超えてから、ターゲット検出または500ユニット到達まで継続
            if (not blue_target_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase10用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 10. 青検出（青ピクセル数1000超えたらphase11へ、最大右モーター400ユニット。条件満たせば右モーター位置記録）
        if phase.get_phase() == 10:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)

            position_limit_reached = abs(current_pos - position_start) >= 400

            if blue_pixel_count > 1000 or position_limit_reached:
                phase.next_phase()
                # phase11用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE2

        # 11. 青ピクセルが500以下まで減るまでcenter追従（500以下でphase12へ、最大右モーター400ユニット。条件満たせば右モーター位置記録）
        if phase.get_phase() == 11:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 400 if self.course_type == "upper" else 1000
            position_limit_reached = abs(current_pos - position_start) >= threshold

            if blue_pixel_count <= 500 or position_limit_reached:
                phase.next_phase()
                # phase12用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE2

        # 12. 右モーター300ユニット移動までcenter追従。300超えたらphase13へ
        if phase.get_phase() == 12:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 300:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2            
            phase.next_phase()

        # 13. 状態リセットしBACK_AND_TURN2へ遷移
        if phase.get_phase() == 13:
            self.reset_action()
            return None, None, Mode.BACK_AND_TURN2

        print("[carry_bottle2_relative] Unexpected state reached.")
        return None, None, Mode.CARRY_BOTTLE2

    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        back_and_turn2の位置判定バージョン。
        実装内容に完全一致:
        0. 左モーター570ユニット移動まで後退（両輪BASE_SPEED）。570超えたらphase1へ、左モーター位置記録。
        1. 右旋回（最低左モーター200ユニットは必ず旋回。200ユニット超えてから一般的な水平黒ライン検出または400ユニット到達まで左:30,右:0で継続。条件満たせばphase2へ）
        2. 終了: 状態リセットしHEAD_GOALへ遷移
        戻り値: (None, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.opposite_course)
        phase = self._phase
        status = self._status

        # 0. 左モーター570ユニット移動まで後退（両輪BASE_SPEED）。570超えたらphase1へ、左モーター位置記録。
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course, status=status)
            if abs(current_pos - position_start) < 570:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.BACK_AND_TURN2
            phase.next_phase()
            # phase1用 左モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.opposite_course, status=status))

        # 1. 右旋回（最低左モーター200ユニットは必ず旋回。200ユニット超えてから一般的な水平黒ライン検出または400ユニット到達まで左:30,右:0で継続。条件満たせばphase2へ）
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 200
            position_limit_reached = position_diff >= 500
            horizontal_line_detected = is_upper_horizontal_line_detected(image)
            # 最低200ユニットは必ず旋回
            if not minimum_position_reached:
                if self.course == "right":
                    return None, (30, 0, 0), Mode.BACK_AND_TURN2
                else:
                    return None, (0, 30, 0), Mode.BACK_AND_TURN2
            # 200ユニット超えてから、水平ライン検出または400ユニット到達まで継続
            if (not horizontal_line_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (30, 0, 0), Mode.BACK_AND_TURN2
                else:
                    return None, (0, 30, 0), Mode.BACK_AND_TURN2
            # 条件を満たしたので次のフェーズへ
            phase.next_phase()

        # 2. 終了: 状態リセットしHEAD_GOALへ遷移
        if phase.get_phase() == 2:
            self.reset_action()
            return None, None, Mode.HEAD_GOAL

        print("[back_and_turn2_relative] Unexpected state reached.")
        return None, None, Mode.BACK_AND_TURN2

    def heading_goal_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        heading_goalの位置判定バージョン。
        実装内容に完全一致:
        0. intersection_y=450で黒水平ライン検出まで中央追従（右モーター距離制限なし）。検出でphase1へ、右モーター位置記録。
        1. 右モーターBの移動距離が300未満なら中央追従、300以上でphase2へ遷移。300到達で右モーター位置記録。
        2. 左旋回（右モーターBの移動距離100未満は常に左旋回。100以上で垂直黒ライン検出開始。600未満の間は左:0,右:30で継続。垂直ライン検出または600到達でphase3へ）
        3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央。青ライン検出時に右モーター位置記録）
        4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        戻り値: (target_x, (左速度, 右速度), モード)
        各行コメントも実装内容と完全一致させること。
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. intersection_y=450で黒水平ライン検出まで中央追従（距離制限なし）。検出でphase1へ、右モーター位置記録。
        if phase.get_phase() == 0:
            if is_lower_horizontal_line_detected(image, intersection_y=450):
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 1. 右モーターBの移動距離が300未満なら中央追従、300以上でphase2へ遷移。300到達で右モーター位置記録。
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            target_x = (self.x1 + self.x2) // 2
            if position_diff < 350:
                return target_x, None, Mode.HEAD_GOAL
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 2. 左旋回（右モーターBの移動距離100未満は常に courseに応じて左:0,右:30または左:30,右:0で左旋回。100以上で垂直黒ライン検出開始。600未満の間は courseに応じて左:0,右:30または左:30,右:0で継続。垂直ライン検出または600到達でphase3へ）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            position_limit_reached = position_diff >= 500
            vertical_line_detected = is_vertical_black_line_detected(image)
            # 常に左旋回。垂直黒ライン検出または600到達でphase3へ
            if (not vertical_line_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.HEAD_GOAL
                else:
                    return None, (30, 0, 0), Mode.HEAD_GOAL
            phase.next_phase()

        # 3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央。青ライン検出時に右モーター位置記録）
        if phase.get_phase() == 3:
            #left_x, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold=80)
            #target_x = left_x if self.course == "right" else right_x
            #if target_x is None:
            #    target_x = (self.x1 + self.x2) // 2
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_limit_reached = abs(current_pos - position_start) >= 600
            if position_limit_reached:
                phase.next_phase()
            else:
                #left_x, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold=80)
                #target_x = left_x if self.course == "right" else right_x
                #if target_x is None:
                #    target_x = (self.x1 + self.x2) // 2
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.HEAD_GOAL

        # 5. 600到達で状態リセットしPAUSE
        if phase.get_phase() == 5:
            self.reset_action()
            return None, None, Mode.PAUSE

        print("[heading_goal_relative] Unexpected state reached.")
        return None, None, Mode.HEAD_GOAL

    def execute_double_loop(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:

        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status
        current_pos = self.get_motor_position(self.course, status=status)

        # phase0: get_blue_line_pixelでBLUE_AREA_THRESHOLD超えたら即phase1へ
        if phase.get_phase() == 0:    
            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase0→phase1: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE0:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE0:
                print(f"[DEBUG] phase0→phase2: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE0}")
                self._phase.next_phase(2)

        # phase1: 青ピクセルが3000未満になったらphase2へ
        if phase.get_phase() == 1:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase1→phase2: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE0:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE0:
                print(f"[DEBUG] phase1→phase3: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE0}")
                self._phase.next_phase()

        # phase2: get_blue_line_pixelでBLUE_AREA_THRESHOLD超えたら即phase3へ（left_pos閾値15000, 左→右エッジ、right_x使用）
        if phase.get_phase() == 2:
            if current_pos < LEFT_POS_THRESHOLD_PHASE0:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase2→phase3: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE2:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE2:
                print(f"[DEBUG] phase2→phase4: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE2}")
                self._phase.next_phase(2)

        # phase3: 青ピクセルが3000未満になったらphase4へ（left_pos閾値15000, 左→右エッジ、right_x使用）
        if phase.get_phase() == 3:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase3→phase4: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE2:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE2:
                print(f"[DEBUG] phase3→phase5: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE2}")
                self._phase.next_phase()

        # phase4: get_blue_line_pixelで18000超えたら即phase5へ（left_pos閾値18000, 左→右エッジ、right_x使用）
        if phase.get_phase() == 4:
            if current_pos < LEFT_POS_THRESHOLD_PHASE2:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase4→phase5: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE4:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE4:
                print(f"[DEBUG] phase4→phase6: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE4}")
                self._phase.next_phase(2)

        # phase5: 青ピクセルが3000未満になったら200距離直進フェーズ(phase6)へ（left_pos閾値18000, 左→右エッジ、right_x使用）
        if phase.get_phase() == 5:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase5→phase6(直進): blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()  # phase6(直進)へ
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            elif current_pos < LEFT_POS_THRESHOLD_PHASE4:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE4:
                print(f"[DEBUG] phase5→phase7: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE4}")
                self._phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase6: 150距離だけ直進するフェーズ。150進んだら次のフェーズへ
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)

            # 150距離進んだら次のフェーズへ
            if abs(current_pos - position_start) >= 150:
                print(f"[DEBUG] phase6(直進)→phase7: 150距離進行完了 (current_pos={current_pos})")
                self._phase.next_phase()
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            return target_x, None, Mode.DOUBLE_LOOP

        # phase7: get_blue_line_pixelで18000超えたら即phase8へ（left_pos閾値21000, 左→右エッジ、right_x使用）
        if phase.get_phase() == 7:
            if current_pos < LEFT_POS_THRESHOLD_PHASE4:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase7→phase8: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE6:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE6:
                print(f"[DEBUG] phase7→phase9: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE6}")
                self._phase.next_phase(2)

        # phase8: get_blue_line_pixelで3000未満になったらphase9へ（left_pos閾値21000, 左→右エッジ）
        if phase.get_phase() == 8:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase8→phase9: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < LEFT_POS_THRESHOLD_PHASE6:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= LEFT_POS_THRESHOLD_PHASE6:
                print(f"[DEBUG] phase8→phase10: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE6}")
                self._phase.next_phase()

        # phase9: 22000到達でCARRY_BOTTLE1へ
        if phase.get_phase() == 9:
            if current_pos < LEFT_POS_THRESHOLD_PHASE8:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            if current_pos >= LEFT_POS_THRESHOLD_PHASE8:
                print(f"[DEBUG] phase9→phase10: current_pos={current_pos} >= {LEFT_POS_THRESHOLD_PHASE8}")
                self._phase.next_phase()

        # phase10: CARRY_BOTTLE1へ
        if phase.get_phase() == 10:
            self.reset_action()
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            return target_x, None, Mode.CARRY_BOTTLE1

        print("[execute_double_loop] Unexpected state reached.")
        return None, None, Mode.DOUBLE_LOOP

