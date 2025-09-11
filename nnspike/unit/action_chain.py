import time  # 時間計測用
from typing import Optional, Tuple  # 型ヒント用

import numpy as np  # 画像処理用

# 定数・モード・ROI設定
from nnspike.constants import OFFSET_Y, ROI_CNN, ROI_LINE_TRACING, Mode, BASE_SPEED, HIGH_SPEED_BASE, ULTRA_HIGH_SPEED, ROI_LINE_HORIZON3, ROI_LOOP, ROI_LINE_CORNER, ROI_COLOR, ROI_LINE_STRAIGHT

# --- 閾値定数（全体で統一管理） ---
BLUE_AREA_MAX_THRESHOLD = 18000
BLUE_AREA_MIN_THRESHOLD = 3000
FIRST_INTERSECTION_LIMIT = 13000
SECOND_INTERSECTION_LIMIT = 17000
THIRD_INTERSECTION_LIMIT = 19000
FOURTH_INTERSECTION_LIMIT = 21000

# ロボット本体クラス
from nnspike.unit.etrobot import ETRobot
# 画像処理・ライン/ターゲット検出関数群
from nnspike.utils.control import (
    find_bottle_center,  # ボトル中心抽出
    get_line_edges_at_y,  # ライン端抽出
    get_virtual_line_target_x,  # 仮想ライン中心抽出
    find_blue_target_center,  # 青ターゲット中心抽出
    get_is_blue_line_at_y,  # 青ライン有無判定
    is_x320_on_blue_target,  # 画像中央付近で青ターゲット抽出
    is_x320_on_red_target,  # 画像中央付近で赤ターゲット抽出
    get_red_target_center_x,  # 赤ターゲット中心抽出
    is_left_black_line_detected,  # 左黒ライン抽出
    is_lower_horizontal_line_detected,  # 下部水平黒ライン抽出
    is_vertical_black_line_detected,  # 垂直黒ライン抽出
    is_upper_horizontal_line_detected,  # 上部水平黒ライン抽出
    get_blue_line_pixel,  # 青オブジェクト面積抽出
    is_fast_corner_detected,  # コーナー抽出
    fill_green_with_white,    # 緑を白で塗りつぶす
    is_center_line_detected,  # 中央ライン検出関数
)

# 型ヒント用: 要素タプル明示
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

    def __init__(self, et: ETRobot, course: str, course_type: str, pid=None) -> None:
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
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN  # 領域定義
        self._init = False
        self.pre_target_x = (self.x1 + self.x2) // 2
        self.pid = pid  # PIDインスタンスを保持（run_manualから共有用）

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
                return None
            return status
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
                    # tupleの中身がint型でない場合は初期値
                    return 0
                else:
                    print(f"[get_motor_position] {motor_key} position invalid: {pos} (return 0)")
                    return 0
            print(f"[get_motor_position] status or motor_key invalid (return 0)")
            return 0
        
        print("[get_motor_position] Unexpected state reached.")
        return 0

    def get_color_sensor_values(self, status):
        """
        カラーセンサーの値（reflected, ambient, color）と黒判定を返す。
        Args:
            status: SpikeStatusオブジェクト（必須）
        Returns:
            dict: {"reflected": int, "ambient": int, "color": int, "is_black": bool}
        """
        if status is None or not hasattr(status, "sensors") or status.sensors is None:
            return {"reflected": 0, "ambient": 0, "color": 0, "is_black": False}
        color = status.sensors.color
        if color is None:
            return {"reflected": 0, "ambient": 0, "color": 0, "is_black": False}
        reflected = color.reflected if hasattr(color, "reflected") and isinstance(color.reflected, int) else 0
        ambient = color.ambient if hasattr(color, "ambient") and isinstance(color.ambient, int) else 0
        color_value = color.color if hasattr(color, "color") and isinstance(color.color, int) else 0
        is_black = (color_value < 100)
        # color_valueのみで排他的な色判定（白・赤青・黒）
        if color_value < 200:
            color_type = "black"
        elif color_value > 900:
            color_type = "white"
        else:
            color_type = "other"
        return {
            "reflected": reflected,
            "ambient": ambient,
            "color": color_value,
            "color_type": color_type
        }

    def get_target_x_by_course(self, image, offset_y, course="right"):
        """
        image, offset_y, course("right"/"left")を受けてtarget_xを返す共通メソッド
        """
        if course == "right":
            _, right_x, _ = get_line_edges_at_y(image, ROI_LINE_TRACING, offset_y, 80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
        elif course == "left":
            left_x, _, _ = get_line_edges_at_y(image, ROI_LINE_TRACING, offset_y, 80)
            target_x = left_x if left_x is not None else (self.x1 + self.x2) // 2
        else:
            target_x = (self.x1 + self.x2) // 2
        return target_x

    def get_target_x_by_course_safe(self, image, course="right"):
        """
        Safe version: Returns target_x for given image, offset_y, and course ("right"/"left").
        Handles None values robustly, no exceptions.
        """
        offset_y = OFFSET_Y
        image = fill_green_with_white(image)
        if course == "right":
            _, right_x, _ = get_line_edges_at_y(image, ROI_LINE_STRAIGHT, offset_y, 80)
            if right_x is not None:
                self.pre_target_x = right_x
                return right_x
            else:
                return self.pre_target_x
        elif course == "left":
            left_x, _, _ = get_line_edges_at_y(image, ROI_LINE_STRAIGHT, offset_y, 80)
            if left_x is not None:
                self.pre_target_x = left_x
                return left_x
            else:
                return self.pre_target_x
        return self.pre_target_x
    
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

        # phase0: ターゲット中心座標へ追従（条件成立で次フェーズへ）
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

        # phase1: 条件成立まで中心座標へ追従、条件成立で次フェーズへ（モーター位置記録）
        if phase.get_phase() == 1:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                phase.next_phase()
                # 次フェーズ用 モーター相対位置記録
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.BLUE_BOTTLE_CATCH

        # phase2: 条件成立後、モーターが所定位置まで中心座標へ追従、到達で次フェーズへ
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
        # phase3: 状態リセットし待機モードへ遷移
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
        左旋回（右モーターBの相対位置差分で判定）。一定値未満の間は旋回、一定値超えたらPAUSE。
        """
        right_position = self.get_motor_position('right', status=status)
        if abs(right_position - phase.get_position_start('position_start')) > 900:
            self.reset_action()
            return None, None, Mode.PAUSE
        return None, (45, 70, 0), Mode.TURN_LEFT_RELATIVE

    def turn_right_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side='left')
        phase = self._phase
        status = self._status
        """
        右旋回（左モーターAの相対位置差分で判定）。一定値未満の間は旋回、一定値超えたらPAUSE。
        """
        left_position = self.get_motor_position('left', status=status)
        if abs(left_position - phase.get_position_start('position_start')) > 430:
            self.reset_action()
            return None, None, Mode.PAUSE
        return None, (30, 0, 0), Mode.TURN_RIGHT_RELATIVE

    def avoid_obstacle(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        障害物回避モードの制御を行う。

        各フェーズで画像処理やモーター移動距離、ライン検出・青面積判定などの条件に応じて、速度・ターゲット座標・モードを返却し、障害物を回避する。
        フェーズ遷移や返却値の詳細は実装内容を参照。
        """

        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.pid.Kp = 1
            self.pid.Ki = 0
            self.pid.Kd = 0
            self.pid.output_limits = (-4, 4)

        phase = self._phase
        status = self._status

        # phase0: 領域検出で次フェーズへ。未検出時は中央追従・回避モード返却
        if phase.get_phase() == 0:
            _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow", roi=ROI_COLOR)
            center_line_detected = is_center_line_detected(image)
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]
            if yellow_pixel_count > 5000:
                print(f"[DEBUG] phase0→phase1: yellow_pixel_count={yellow_pixel_count} > 5000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            elif center_line_detected:
                if color_type != "white":
                    self.pid.Kp = 1
                    self.pid.Ki = 0
                    self.pid.Kd = 0
                    self.pid.output_limits = (-4, 4)
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                    return target_x, (0, 0, ULTRA_HIGH_SPEED), Mode.AVOID_OBSTACLE
                else:
                    self.pid.Kp = 5
                    self.pid.Ki = 0
                    self.pid.Kd = 0
                    self.pid.output_limits = (-8, 8)
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                    return target_x, (0, 0, HIGH_SPEED_BASE), Mode.AVOID_OBSTACLE
            else:
                # Lock if passed once and now False
                self.pid.Kp = 20
                self.pid.Ki = 0
                self.pid.Kd = 5
                self.pid.output_limits = (-20, 20)
                image = fill_green_with_white(image)
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase1: 領域検出で次フェーズへ。未検出時は中心または中央追従・回避モード返却
        if phase.get_phase() == 1:
            yellow_cx, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow", roi=ROI_COLOR)
            if yellow_pixel_count > 18000:
                print(f"[DEBUG] phase1→phase2: yellow_pixel_count={yellow_pixel_count} yellow_cx={yellow_cx} > 18000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if yellow_cx is not None:
                    target_x = yellow_cx[0]
                else:
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase2: 左旋回（条件成立まで速度調整、到達で次フェーズへ・モーター位置記録）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 300:
                if self.course == "right":
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase2→phase3: position_diff={position_diff} current_pos={current_pos} >= 300")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase3: 直進（条件成立まで定速走行、到達で次フェーズへ・モーター位置記録）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 200:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase3→phase4: position_diff={position_diff} current_pos={current_pos} >= 200")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.opposite_course, status=status))

        # phase4: 条件成立まで定速走行、成立で判定・次フェーズへ（モーター位置記録）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 450:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
                
            if is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3):
                print(f"[DEBUG] phase4→phase5: position_diff={position_diff} current_pos={current_pos} (horizontal line detected)")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE

        # phase5: 右モーター移動距離が所定値未満なら中央追従、以上で次フェーズへ、右モーター位置記録。
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]
            # 距離450に到達する、もしくはcolor_typeが白以外になったら次フェーズ
            if position_diff >= 450 or color_type != "white":
                print(f"[DEBUG] phase5→phase6: position_diff={position_diff} current_pos={current_pos} color_type={color_type} color_value={color_info['color']}")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.AVOID_OBSTACLE

        # phase6: 左旋回（短距離は低速、長距離は高速、所定値以上で次フェーズへ。所定値未満かつ垂直黒ライン検出で次フェーズへ）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            distance = abs(current_pos - position_start)
            if distance < 350:
                vertical_detected = is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=150)
                if vertical_detected:
                    print(f"[DEBUG] phase6→phase7: distance={distance} current_pos={current_pos} (vertical black line detected)")
                    phase.next_phase()
                    phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
                    self.pid.Kp = 50
                    self.pid.Ki = 0
                    self.pid.Kd = 5
                    self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                    return None, None, Mode.AVOID_OBSTACLE
                else:
                    if self.course == "right":
                        return None, (5, 35, 0), Mode.AVOID_OBSTACLE
                    else:
                        return None, (35, 5, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase6→phase7: distance={distance} current_pos={current_pos} (distance >= 400, next phase)")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
                self.pid.Kp = 50
                self.pid.Ki = 0
                self.pid.Kd = 5
                self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                return None, None, Mode.AVOID_OBSTACLE

        # phase7: Go to next phase when distance threshold is reached. Otherwise, return get_target_x_by_course with AVOID_OBSTACLE.
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)

            # if is_fast_corner_detected(image, course=self.course):
            #     print(f"[DEBUG] phase7: is_fast_corner_detected=True at current_pos={current_pos}, course={self.course}")

            if position_diff >= 1500:
                print(f"[DEBUG][phase7→phase8] Distance threshold reached: position_diff={position_diff}, current_pos={current_pos}, threshold=1500 → phase8")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase8: After right motor travels threshold, check vertical black line to go to next phase. Otherwise, return get_target_x_by_course with AVOID_OBSTACLE.
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            center_line_detected = is_center_line_detected(image)
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]

            # if is_fast_corner_detected(image, course=self.course):
            #     print(f"[DEBUG] phase8: is_fast_corner_detected=True at current_pos={current_pos}, course={self.course}")

            if position_diff > 2600:
                print(f"[DEBUG][phase8→phase9] Right motor distance: position_diff={position_diff}, current_pos={current_pos} > 2600 → phase9")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            elif center_line_detected:
                if color_type != "white":
                    self.pid.Kp = 1
                    self.pid.Ki = 0
                    self.pid.Kd = 0
                    self.pid.output_limits = (-4, 4)
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                    return target_x, (0, 0, ULTRA_HIGH_SPEED), Mode.AVOID_OBSTACLE
                else:
                    self.pid.Kp = 5
                    self.pid.Ki = 0
                    self.pid.Kd = 0
                    self.pid.output_limits = (-8, 8)
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                    return target_x, (0, 0, HIGH_SPEED_BASE), Mode.AVOID_OBSTACLE
            else:
                # Lock if passed once and now False
                self.pid.Kp = 50
                self.pid.Ki = 0
                self.pid.Kd = 5
                self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                image = fill_green_with_white(image)
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase9: Go to DOUBLE_LOOP if blue area threshold or right motor distance is reached, otherwise continue AVOID_OBSTACLE.
        if phase.get_phase() == 9:
            blue_area = get_blue_line_pixel(image)
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                current_pos = self.get_motor_position(self.course, status=status)
                print(f"[DEBUG][phase9→DOUBLE_LOOP] Blue area: blue_area={blue_area} > threshold → DOUBLE_LOOP, current_pos={current_pos}")
                self.reset_action()
                return target_x, (0, 0, BASE_SPEED), Mode.DOUBLE_LOOP
            else:
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        print("[avoid_obstacle] Unexpected state reached.")
        return None, None, Mode.AVOID_OBSTACLE

    def carry_bottle1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        carry_bottle1の位置判定バージョン。
        実装内容に完全一致:
        0. 右エッジトレース（赤ピクセル数が閾値を超えたらphase1へ、右モーター初期位置記録）
        1. 赤ボトル中心追従（右モーター相対位置差分が一定値未満の間、赤ピクセルが条件を満たせばcenter、満たさなければ中央。一定値を超えたらphase2へ、右モーター位置記録）
        2. 左エッジトレース（コース種別に応じた閾値未満の間、閾値を超えたらphase3へ、右モーター位置記録）
        3. 左旋回（右モーター相対位置差分が一定値未満の間旋回。一定値超えたらphase4へ、右モーター位置記録）
        4. 直進（右モーターが所定値移動まで。所定値超えたらphase5へ、右モーター位置記録、pre_target_x初期化）
        5. 仮想ライン直進（右モーターが所定値移動まで仮想ライン中心座標取得処理、pre_target_x更新。所定値超えたらphase6へ、右モーター位置記録）
        6. 直進（右モーターが所定値移動まで。所定値超えたらphase7へ、右モーター位置記録）
        7. 左旋回（is_x320_on_blue_targetがTrueになるまで旋回、最低回転量・最大回転量はコース種別で異なる。条件満たせばphase8へ、右モーター位置記録）
        8. 青検出（青ピクセル数が閾値を超えたらphase9へ）
        9. 青ピクセル数が閾値以上の間center追従、閾値以下でphase10へ、右モーター位置記録
        10. 青ピクセル数が閾値以下になってから右モーターが所定値移動までcenter追従。条件を満たしたらphase11へ
        11. 状態リセットしBACK_AND_TURN1へ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 右エッジトレース（赤ピクセル数が一定値を超えたらphase1へ、右モーター初期位置記録）
        if phase.get_phase() == 0:
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            _, _, red_pixel_count = find_bottle_center(image=image, color="red", roi=ROI_COLOR)
            if red_pixel_count > 3000:
                print(f"[DEBUG] phase0→phase1: red_pixel_count={red_pixel_count} > 3000")
                phase.next_phase()
                # phase1用 右モーター相対位置記録（絶対値）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 1. 赤ボトル中心追従（右モーター相対位置差分が一定値未満の間、赤ピクセルが条件を満たせばcenter、満たさなければ中央。一定値を超えたらphase2へ、右モーター位置記録）
        if phase.get_phase() == 1:
            center, _, red_px = find_bottle_center(image=image, color="red", roi=ROI_COLOR)
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            # 右モーター相対位置差分で継続判定（一定値未満の間、赤ピクセルが条件を満たせばcenter、満たさなければ中央）
            if abs(current_pos - position_start) < 1000:
                if center is not None and red_px is not None and red_px >= 500:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE1
            # 一定値を超えたら次フェーズへ
            print(f"[DEBUG] phase1→phase2: position_diff={abs(current_pos - position_start)} >= 1000")
            phase.next_phase()
            # phase2用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 2. 左エッジトレース（コース種別に応じた閾値未満の間、閾値を超えたらphase3へ、右モーター位置記録）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 1300 if self.course_type == "upper" else 700
            if abs(current_pos - position_start) < threshold:
                target_x = self.get_target_x_by_course(image, offset_y=300, course=self.opposite_course)
                return target_x, None, Mode.CARRY_BOTTLE1
            # 閾値を超えたら次フェーズへ
            print(f"[DEBUG] phase2→phase3: position_diff={abs(current_pos - position_start)} >= {threshold}")
            phase.next_phase()
            # phase3用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 3. 左旋回（右モーター相対位置差分が一定値未満の間旋回。一定値超えたらphase4へ、右モーター位置記録）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 390:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE1
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] phase3→phase4: position_diff={abs(current_pos - position_start)} >= 390")
            phase.next_phase()
            # phase4用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 4. 直進（右モーターが一定値移動まで。一定値超えたらphase5へ、右モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 200:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] phase4→phase5: position_diff={abs(current_pos - position_start)} >= 200")
            phase.next_phase()
            # phase5用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            self.pre_target_x = (self.x1 + self.x2) // 2

        # 5. 仮想ライン直進（右モーターが一定値移動まで仮想ライン中心座標取得処理、pre_target_x更新。一定値超えたらphase6へ、右モーター位置記録）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 1500:
                # 仮想ライン中心座標取得処理
                temp_x = get_virtual_line_target_x(image, previous_center_x=self.pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    self.pre_target_x = temp_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    self.pre_target_x = target_x
                return target_x, None, Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] phase5→phase6: position_diff={abs(current_pos - position_start)} >= 1500")
            phase.next_phase()
            # phase6用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 6. 直進（右モーターが一定値移動まで。一定値超えたらphase7へ、右モーター位置記録）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            if abs(current_pos - position_start) < 1300:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] phase6→phase7: position_diff={abs(current_pos - position_start)} >= 1300")
            phase.next_phase()
            # phase7用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 7. 左旋回（is_x320_on_blue_targetがTrueになるまで旋回、最低回転量・最大回転量はコース種別で異なる。条件満たせばphase8へ、右モーター位置記録）
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
            # 最低回転量後にblue_target検出、または最大回転量到達で次へ
            if (minimum_rotation_done and blue_target_detected) or position_limit_reached:
                if self.course_type == "lower":
                    print(f"[DEBUG] phase7→phase9: minimum_rotation_done={minimum_rotation_done}, blue_target_detected={blue_target_detected}, position_limit_reached={position_limit_reached}")
                    phase.next_phase(skip=2)  # スキップ
                else:
                    print(f"[DEBUG] phase7→phase8: minimum_rotation_done={minimum_rotation_done}, blue_target_detected={blue_target_detected}, position_limit_reached={position_limit_reached}")
                    phase.next_phase()
            else:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE1
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE1

        # 8. 青検出（青ピクセル数が一定値を超えたらphase9へ）
        if phase.get_phase() == 8:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                print(f"[DEBUG] phase8→phase9: blue_pixel_count={blue_pixel_count} > 1000")
                phase.next_phase()
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 9. 青ピクセル数が一定値以上の間center追従、一定値以下でphase10へ、右モーター位置記録
        if phase.get_phase() == 9:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 300:
                print(f"[DEBUG] phase9→phase10: blue_pixel_count={blue_pixel_count} <= 300")
                phase.next_phase()
                # phase10用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 10. 青が一定値以下になってから右モーターが一定値移動までcenter追従。条件を満たしたら次のphaseへ
        if phase.get_phase() == 10:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 300
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]
            position_diff = abs(current_pos - position_start)
            if position_diff >= threshold or color_type == "other":
                print(f"[DEBUG] phase10→phase11: position_diff={position_diff} >= {threshold} or color_type={color_type}")
                phase.next_phase()
            else:
                center, _, _ = find_blue_target_center(image)
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE1
            
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
        0. 後退（右モーターが所定値移動まで。所定値超えたらphase1へ、右モーター位置記録）
        1. 左旋回（最低回転量は必ず旋回。最低回転量超えてからターゲット検出または最大回転量到達まで旋回。条件満たせばphase2へ）
        2. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        戻り値: (None, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 後退（コース側モーターが所定値移動まで。所定値超えたらphase1へ、モーター位置記録）
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 600:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.BACK_AND_TURN1
            print(f"[DEBUG] phase0→phase1: position_diff={position_diff} >= 600")
            phase.next_phase()
            # phase1用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 1. 左旋回（最低回転量は必ず旋回。最低回転量超えてからターゲット検出または最大回転量到達まで旋回。条件満たせばphase2へ）
        if phase.get_phase() == 1:
            red_target_detected = is_x320_on_red_target(image, x_tolerance=80)
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 450
            position_limit_reached = position_diff >= 940
            # 最低回転量は必ず旋回
            if not minimum_position_reached:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.BACK_AND_TURN1
                else:
                    return None, (30, 0, 0), Mode.BACK_AND_TURN1
            # 最低回転量超えてから、ターゲット検出または最大回転量到達まで継続
            if (not red_target_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.BACK_AND_TURN1
                else:
                    return None, (30, 0, 0), Mode.BACK_AND_TURN1
            print(f"[DEBUG] phase1→phase2: minimum_position_reached={minimum_position_reached}, red_target_detected={red_target_detected}, position_limit_reached={position_limit_reached}")
            phase.next_phase()
            return None, None, Mode.BACK_AND_TURN1

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
        0. 赤ターゲット中心追従（青ピクセル数が閾値未満の間は赤中心追従、閾値以上でphase1へ）
        1. 青ボトル中心追従（青ピクセル数が閾値以上の間center追従、閾値未満でphase2へ、右モーター位置記録）
        2. 右モーターが所定値移動までcenter追従。所定値超えたらphase3へ、右モーター位置記録
        3. 左黒ライン検出まで左旋回。最大回転量まで。検出または最大回転量超えたらphase4へ、右モーター位置記録
        4. 直進（右モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase5へ、右モーター位置記録）
        5. 左旋回（右モーターが所定値移動まで旋回。所定値超えたらphase6へ、右モーター位置記録）
        6. 直進（右モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase7へ、右モーター位置記録、pre_target_x初期化）
        7. 仮想ライン直進（右モーターが所定値移動まで仮想ライン中心座標取得処理、pre_target_x更新。所定値超えたらphase8へ、右モーター位置記録）
        8. 直進（右モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase9へ、右モーター位置記録）
        9. 左旋回（青ターゲット検出まで、最低回転量・最大回転量。条件満たせばphase10へ、右モーター位置記録）
        10. 青検出（青ピクセル数が閾値を超えたらphase11へ、最大回転量。条件満たせば右モーター位置記録）
        11. 青ピクセルが閾値以下まで減るまでcenter追従（閾値以下でphase12へ、最大回転量。条件満たせば右モーター位置記録）
        12. 右モーターが所定値移動までcenter追従。所定値超えたらphase13へ
        13. 状態リセットしBACK_AND_TURN2へ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 赤ターゲット中心追従（青ピクセル数が閾値未満の間は赤中心追従、閾値以上で次フェーズへ）
        if phase.get_phase() == 0:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
            if blue_pixel_count < 18000:
                # 赤ターゲット中心追従
                red_center_x = get_red_target_center_x(image)
                # 赤センター最優先
                if red_center_x is not None:
                    target_x = red_center_x
                elif blue_pixel_count >= 5000 and center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                print(f"[DEBUG] phase0→phase1: blue_pixel_count={blue_pixel_count} >= 18000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 1. 青ボトル中心追従（青ピクセル数が十分な間はcenter追従、少なくなったらphase2へ移行し右モーター位置記録）
        if phase.get_phase() == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
            target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
            # 上限距離チェック
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]
            if position_diff >= 1000 or blue_pixel_count < 5000 or color_type == "other":
                print(f"[DEBUG] phase1→phase2: position_diff={position_diff} >= 1000 or blue_pixel_count={blue_pixel_count} < 5000 or color_type={color_type}")
                phase.next_phase()
                # phase2用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                # 青ボトル中心x座標へ追従
                return target_x, None, Mode.CARRY_BOTTLE2

        # 2. 右モーターが所定値移動までcenter追従。所定値超えたら次フェーズへ、右モーター位置記録
        if phase.get_phase() == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 170:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase2→phase3: position_diff={position_diff} >= 200")
            phase.next_phase()
            # phase3用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 3. 左黒ライン検出まで左旋回。最大回転量まで。検出または最大回転量到達で次フェーズへ、右モーター位置記録
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            min_limit = 700
            max_limit = 1200
            position_delta = abs(current_pos - position_start)
            position_limit_reached = position_delta >= max_limit

            # 最小閾値未満は検知開始しない
            if position_delta < min_limit and not position_limit_reached:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2

            # 最小閾値以上になったら判定開始
            line_detected = is_left_black_line_detected(image, self.course)

            if (not line_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase3→phase4: line_detected={line_detected}, position_delta={position_delta} >= {min_limit}, position_limit_reached={position_limit_reached}")
            phase.next_phase()
            # phase4用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 4. 直進（コース側モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase5へ、モーター位置記録）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 870 if self.course_type == "upper" else 1300
            position_diff = abs(current_pos - position_start)
            if position_diff < threshold:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase4→phase5: position_diff={position_diff} >= {threshold}")
            phase.next_phase()
            # phase5用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 5. 左旋回（コース側モーターが所定値移動まで、courseに応じて旋回方向決定。所定値超えたらphase6へ、モーター位置記録）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 350:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase5→phase6: position_diff={position_diff} >= 350")
            phase.next_phase()
            # phase6用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 6. 直進（コース側モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase7へ、モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 100:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase6→phase7: position_diff={position_diff} >= 100")
            phase.next_phase()
            # phase7用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            self.pre_target_x = (self.x1 + self.x2) // 2

        # 7. 仮想ライン直進（コース側モーターが所定値移動まで仮想ライン中心座標取得処理、pre_target_x更新。所定値超えたらphase8へ、モーター位置記録）
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 1400:
                # 仮想ライン中心座標取得処理
                temp_x = get_virtual_line_target_x(image, previous_center_x=self.pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    self.pre_target_x = temp_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    self.pre_target_x = target_x
                return target_x, None, Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase7→phase8: position_diff={position_diff} >= 1400")
            phase.next_phase()
            # phase8用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 8. 直進（コース側モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase9へ、モーター位置記録）
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 400:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase8→phase9: position_diff={position_diff} >= 400")
            phase.next_phase()
            # phase9用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 9. 左旋回（青ターゲット検出まで、最低回転量・最大回転量。条件満たせばphase10へ、右モーター位置記録）
        if phase.get_phase() == 9:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 300
            position_limit_reached = position_diff >= 500
            # 最低回転量は必ず旋回
            if not minimum_position_reached:
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            # 最低回転量超えてから、ターゲット検出または最大回転量到達まで継続
            if (not blue_target_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.CARRY_BOTTLE2
                else:
                    return None, (30, 0, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] phase9→phase10: minimum_position_reached={minimum_position_reached}, blue_target_detected={blue_target_detected}, position_limit_reached={position_limit_reached}")
            phase.next_phase()
            # phase10用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # 10. 青検出（青ピクセル数が一定値を超えたらphase11へ、最大回転量。条件満たせば右モーター位置記録）
        if phase.get_phase() == 10:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            position_limit_reached = position_diff >= 400
            if blue_pixel_count > 1000 or position_limit_reached:
                print(f"[DEBUG] phase10→phase11: blue_pixel_count={blue_pixel_count} > 1000 or position_diff={position_diff} >= 400")
                phase.next_phase()
                # phase11用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE2

        # 11. 青ピクセルが一定値以下まで減るまでcenter追従（一定値以下でphase12へ、最大回転量。条件満たせば右モーター位置記録）
        if phase.get_phase() == 11:
            center, _, blue_pixel_count = find_blue_target_center(image)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            threshold = 400 if self.course_type == "upper" else 1000
            position_diff = abs(current_pos - position_start)
            position_limit_reached = position_diff >= threshold
            if blue_pixel_count <= 300 or position_limit_reached:
                print(f"[DEBUG] phase11→phase12: blue_pixel_count={blue_pixel_count} <= 300 or position_diff={position_diff} >= {threshold}")
                phase.next_phase()
                # phase12用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE2

        # 12. コース側モーターが所定値移動までcenter追従。所定値超えたらphase13へ
        if phase.get_phase() == 12:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]
            if position_diff >= 300 or color_type == "other":
                print(f"[DEBUG] phase12→phase13: position_diff={position_diff} >= 300 or color_type={color_type}")
                phase.next_phase()
            else:
                center, _, _ = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2

        # 13. 状態リセットしBACK_AND_TURN2へ遷移
        if phase.get_phase() == 13:
            self.reset_action()
            return None, None, Mode.BACK_AND_TURN2

        print("[carry_bottle2_relative] Unexpected state reached.")
        return None, None, Mode.CARRY_BOTTLE2

    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        back_and_turn2の位置判定バージョン。
        実装内容:
        0. 左モーターが所定位置まで後退（両輪定速）。条件成立で次フェーズへ、モーター位置記録。
        1. 右旋回（所定位置までは必ず旋回。条件成立後、ライン検出または所定位置到達まで所定速度で継続。条件成立で次フェーズへ）
        2. 終了: 状態リセットし目標モードへ遷移
        戻り値: (None, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.opposite_course)
        phase = self._phase
        status = self._status

        # 0. 左モーターが抽象的な基準位置まで後退（両輪定速）。条件成立で次フェーズへ、モーター位置記録。
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 570:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.BACK_AND_TURN2
            print(f"[DEBUG] phase0→phase1: position_diff={position_diff} >= 570")
            phase.next_phase()
            # phase1用 モーター相対位置記録（抽象化）
            phase.set_position_start("position_start", self.get_motor_position(self.opposite_course, status=status))

        # 1. 右旋回（抽象的な基準位置までは必ず旋回。条件成立後、ライン検出または基準位置到達まで所定速度で継続。条件成立で次フェーズへ）
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course, status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 200
            position_limit_reached = position_diff >= 500
            horizontal_line_detected = is_upper_horizontal_line_detected(image)
            # 抽象的な基準位置までは必ず旋回
            if not minimum_position_reached:
                if self.course == "right":
                    return None, (30, 0, 0), Mode.BACK_AND_TURN2
                else:
                    return None, (0, 30, 0), Mode.BACK_AND_TURN2
            # 基準位置到達後、ライン検出または別基準位置到達まで継続
            if (not horizontal_line_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (30, 0, 0), Mode.BACK_AND_TURN2
                else:
                    return None, (0, 30, 0), Mode.BACK_AND_TURN2
            print(f"[DEBUG] phase1→phase2: minimum_position_reached={minimum_position_reached}, horizontal_line_detected={horizontal_line_detected}, position_limit_reached={position_limit_reached}")
            # 条件を満たしたので次のフェーズへ
            phase.next_phase()

        # 2. 終了: 状態リセットし目標モードへ遷移（抽象化）
        if phase.get_phase() == 2:
            self.reset_action()
            return None, None, Mode.HEAD_GOAL

        print("[back_and_turn2_relative] Unexpected state reached.")
        return None, None, Mode.BACK_AND_TURN2

    def heading_goal_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        heading_goalの位置判定バージョン。
        実装内容に完全一致:
        0. 水平ライン検出まで中央追従（移動距離制限なし）。検出でphase1へ、右モーター位置記録。
        1. 中央追従から左旋回へ遷移判定（移動距離による遷移制御）
        2. 左旋回（左旋回継続。垂直黒ライン検出または移動距離上限到達でphase3へ）
        3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央。青ライン検出時に右モーター位置記録）
        4. 青ライン検出後、移動距離制限内で左エッジトレース、制限到達でPAUSE（状態リセット）
        戻り値: (target_x, (左速度, 右速度), モード)
        各行コメントも実装内容と完全一致させること。
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # 0. 所定のintersection_yで黒水平ライン検出まで中央追従（距離制限なし）。検出でphase1へ、右モーター位置記録。
        if phase.get_phase() == 0:
            if is_lower_horizontal_line_detected(image, intersection_y=450):
                print(f"[DEBUG] phase0→phase1: horizontal_line_detected at intersection_y=450")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 1. コース側モーターの移動距離が所定値未満なら中央追従、所定値以上で次フェーズへ遷移。到達でモーター位置記録。
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            color_info = self.get_color_sensor_values(status)
            color_type = color_info["color_type"]
            if position_diff >= 350 or color_type != "white":
                print(f"[DEBUG] phase1→phase2: position_diff={position_diff} current_pos={current_pos} >= 350 or color_type={color_type}")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.HEAD_GOAL

        # 2. 左旋回（courseに応じて左旋回。垂直黒ライン検出または移動距離上限到達でphase3へ）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            position_limit_reached = position_diff >= 500
            vertical_line_detected = is_vertical_black_line_detected(image)
            # Continue turning left. If vertical black line detected or position limit reached, go to phase3
            if (not vertical_line_detected) and (not position_limit_reached):
                if self.course == "right":
                    return None, (0, 30, 0), Mode.HEAD_GOAL
                else:
                    return None, (30, 0, 0), Mode.HEAD_GOAL
            print(f"[DEBUG] phase2→phase3: vertical_line_detected={vertical_line_detected} position_diff={position_diff} >= 500")
            phase.next_phase()

        # 3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央。青ライン検出時に右モーター位置記録）
        if phase.get_phase() == 3:
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                print(f"[DEBUG] phase3→phase4: blue_line={blue_line}")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、コース側モーターの移動距離が所定値未満の間は左エッジトレース、到達でPAUSE（状態リセット）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_limit_reached = abs(current_pos - position_start) >= 300
            if position_limit_reached:
                print(f"[DEBUG] phase4→phase5: position_diff={abs(current_pos - position_start)} current_pos={current_pos} >= 300")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.HEAD_GOAL

        # 5. 青ライン検出後、コース側モーターの移動距離が所定値未満の間は直進、到達で次のフェーズへ
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_limit_reached = abs(current_pos - position_start) >= 200  # straight distance check
            if position_limit_reached:
                print(f"[DEBUG] phase5→phase6: position_diff={abs(current_pos - position_start)} current_pos={current_pos} >= 200")
                phase.next_phase()
            else:
                # go straight
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.HEAD_GOAL

        # 6. 所定距離到達で状態リセットしPAUSE
        if phase.get_phase() == 6:
            self.reset_action()
            return None, None, Mode.PAUSE

        print("[heading_goal_relative] Unexpected state reached.")
        return None, None, Mode.HEAD_GOAL

    def execute_double_loop(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        double loopの制御。
        実装内容:
        0. 青領域が所定値を超えたら即次フェーズへ。未満なら中央追従または次フェーズ。
        1. 青領域が所定値未満になったら次フェーズへ。
        ...（以降のフェーズは既存処理に準ずる）
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        """
        double loopの制御。
        実装内容:
        0. 青領域が所定値を超えたら即次フェーズへ。未満なら中央追従または次フェーズ。
        1. 青領域が所定値未満になったら次フェーズへ。
        ...（以降のフェーズは既存処理に準ずる）
        戻り値: (target_x, (左速度, 右速度), モード)
        """

        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.pid.Kp = 50
            self.pid.Ki = 0
            self.pid.Kd = 5
            self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)

        phase = self._phase
        status = self._status
        current_pos = self.get_motor_position(self.course, status=status)

        # phase0: 青領域が条件を超えたら即次フェーズへ
        if phase.get_phase() == 0:    
            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase0→phase1: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < FIRST_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= FIRST_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase0→phase2: current_pos={current_pos} >= {FIRST_INTERSECTION_LIMIT}")
                self._phase.next_phase(2)

        # phase1: 青領域が条件未満になったら次フェーズへ
        if phase.get_phase() == 1:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase1→phase2: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < FIRST_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= FIRST_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase1→phase3: current_pos={current_pos} >= {FIRST_INTERSECTION_LIMIT}")
                self._phase.next_phase()

        # phase2: get_blue_line_pixelでBLUE_AREA_THRESHOLD超えたら即phase3へ（left_pos閾値15000, 左→右エッジ、right_x使用）
        if phase.get_phase() == 2:
            if current_pos < FIRST_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase2→phase3: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < SECOND_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= SECOND_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase2→phase4: current_pos={current_pos} >= {SECOND_INTERSECTION_LIMIT}")
                self._phase.next_phase(2)

        # phase3: 青領域が条件未満になったら次フェーズへ（抽象化）
        if phase.get_phase() == 3:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase3→phase4: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < SECOND_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= SECOND_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase3→phase5: current_pos={current_pos} >= {SECOND_INTERSECTION_LIMIT}")
                self._phase.next_phase()

        # phase4: 青領域が条件を超えたら即次フェーズへ（抽象化）
        if phase.get_phase() == 4:
            if current_pos < SECOND_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase4→phase5: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < THIRD_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= THIRD_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase4→phase6: current_pos={current_pos} >= {THIRD_INTERSECTION_LIMIT}")
                self._phase.next_phase(2)

        # phase5: 青領域が条件未満になったら直進フェーズへ（抽象化）
        if phase.get_phase() == 5:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase5->phase6(straight): blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()  # phase6(直進)へ
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            elif current_pos < THIRD_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= THIRD_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase5→phase7: current_pos={current_pos} >= {THIRD_INTERSECTION_LIMIT}")
                self._phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase6: 所定距離だけ直進するフェーズ。条件成立で次のフェーズへ（抽象化）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)

            # 所定距離進んだら次のフェーズへ
            if abs(current_pos - position_start) >= 150:
                print(f"[DEBUG] phase6->phase7(straight): 150 units moved (current_pos={current_pos})")
                self._phase.next_phase()
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            return target_x, None, Mode.DOUBLE_LOOP

        # phase7: 青領域が条件を超えたら即次フェーズへ（抽象化）
        if phase.get_phase() == 7:
            if current_pos < THIRD_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] phase7→phase8: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < FOURTH_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= FOURTH_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase7→phase9: current_pos={current_pos} >= {FOURTH_INTERSECTION_LIMIT}")
                self._phase.next_phase(2)

        # phase8: 青領域が条件未満になったら次フェーズへ（抽象化）
        if phase.get_phase() == 8:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] phase8→phase9: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD}")
                self._phase.next_phase()
            elif current_pos < FOURTH_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.DOUBLE_LOOP
            elif current_pos >= FOURTH_INTERSECTION_LIMIT:
                print(f"[DEBUG] phase8→phase10: current_pos={current_pos} >= {FOURTH_INTERSECTION_LIMIT}")
                self._phase.next_phase()

        # phase9: 条件未満ならDOUBLE_LOOP継続、条件到達で次モードへ（抽象化）
        if phase.get_phase() == 9:
            if current_pos < FOURTH_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, None, Mode.DOUBLE_LOOP
            else:   
                print(f"[DEBUG] phase9→phase10: current_pos={current_pos} >= {FOURTH_INTERSECTION_LIMIT}")
                self._phase.next_phase()

        # phase10: CARRY_BOTTLE1へ
        if phase.get_phase() == 10:
            self.reset_action()
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            return target_x, None, Mode.CARRY_BOTTLE1

        print("[execute_double_loop] Unexpected state reached.")
        return None, None, Mode.DOUBLE_LOOP

