
import time
from typing import Optional, Tuple
import numpy as np
from nnspike.unit.etrobot import ETRobot
from nnspike.constants import HIGH_SPEED_BASE
from nnspike.unit.action_chain import PhaseManager
from nnspike.constants import ROI_CNN

class FastLapChain(object):

    """
    最速ラップタイムを記録するためのアクションチェーン管理クラス。
    ActionChainの設計・フェーズ管理を踏襲。
    """
    def __init__(self, et: ETRobot, course: str, course_type: str = "upper", pid=None) -> None:
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
        self.pid = pid  # PIDはNoneでもOK

    def initialize_action(self, motor_side: str = "right"):
        """アクション開始時の状態初期化処理.

        motor_side: "right"または"left"で初期位置記録対象を指定する。
        """
        self._phase = PhaseManager()
        self._phase.set_position_start("position_start", self.get_motor_position(motor_side))
        self._init = True

    def reset_action(self):
        """アクション終了時の状態リセット処理."""
        self._init = False

    def get_motor_position(self, motor_side: str = "right") -> int:
        """
        ETRobotのget_motor_relative_positionを直接呼び出す。
        """
        return self.et.get_motor_relative_position(motor_side)

    def get_color_sensor_values(self) -> dict:
        """
        ETRobotのget_color_sensorを直接呼び出し、colorとcolor_typeのみ返す。
        """
        color_value, color_type = self.et.get_color_sensor()
        return {
            "color": color_value,
            "color_type": color_type
        }

    def turn_left_yaw(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int, int]], 'Mode']:
        # ActionChain設計に厳密に合わせる: self._init判定→initialize_action→phase管理
        from nnspike.constants import Mode
        if not self._init:
            self.initialize_action(motor_side='right')
            self.et.set_start_yaw()
        phase = self._phase
        et = self.et

        # phase0: 旋回中（yaw判定）
        if phase.get_phase() == 0:
            stop_turn = et.is_yaw_turn_finished(side="left", threshold_deg=90.0)
            if stop_turn:
                phase.next_phase()
                print(f"[TURN_LEFT] reached -90 deg and stopped | yaw={et.get_yaw():.2f}")
                return None, (0, 0, 0), Mode.TURN_LEFT_YAW
            else:
                print(f"[TURN_LEFT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                return None, (-30, 30, 0), Mode.TURN_LEFT_YAW

        # phase1: 微調整（±5度以内1秒でPAUSE）
        if phase.get_phase() == 1:
            if not hasattr(self, 'turn_left_reference_yaw') or self.turn_left_reference_yaw is None:
                self.turn_left_reference_yaw = et.get_yaw()
                self.turn_left_adjust_timer = time.time()
                self.turn_left_in_tolerance_time = None
            error = et.get_yaw() - self.turn_left_reference_yaw
            elapsed = time.time() - self.turn_left_adjust_timer if self.turn_left_adjust_timer is not None else 0
            if abs(error) <= 5.0:
                if self.turn_left_in_tolerance_time is None:
                    self.turn_left_in_tolerance_time = time.time()
                tolerance_elapsed = time.time() - self.turn_left_in_tolerance_time
                print(f"[TURN_LEFT][ADJUST][TOLERANCE] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, tolerance_elapsed={tolerance_elapsed:.2f}s")
                if tolerance_elapsed >= 1.0:
                    print(f"[TURN_LEFT][ADJUST][STOP] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, tolerance_elapsed={tolerance_elapsed:.2f}s")
                    self._phase = None
                    self.turn_left_reference_yaw = None
                    self.turn_left_adjust_timer = None
                    self.turn_left_in_tolerance_time = None
                    return None, (0, 0, 0), Mode.PAUSE
            else:
                self.turn_left_in_tolerance_time = None
                if elapsed < 1.0:
                    if error < 0:
                        print(f"[TURN_LEFT][ADJUST] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                        return None, (15, -15, 0), Mode.TURN_LEFT_YAW
                    else:
                        print(f"[TURN_LEFT][ADJUST] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                        return None, (-15, 15, 0), Mode.TURN_LEFT_YAW
                else:
                    print(f"[TURN_LEFT][ADJUST][TIMEOUT] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                    self._phase = None
                    self.turn_left_reference_yaw = None
                    self.turn_left_adjust_timer = None
                    self.turn_left_in_tolerance_time = None
                    return None, (0, 0, 0), Mode.TURN_LEFT_YAW

        # 速度返却（NoneでOK、et.set_motor_speedで直接制御）
        return None, None, Mode.TURN_LEFT_YAW

    def turn_right_yaw(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int, int]], 'Mode']:
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side='left')
        phase = self._phase
        """
        右旋回（左モーターAの相対位置差分で判定）。一定値未満の間は旋回、一定値超えたらPAUSE。
        """
        left_position = self.get_motor_position('left')
        if abs(left_position - phase.get_position_start('position_start')) > 430:
            self.reset_action()
            return None, None, Mode.PAUSE
        return None, (30, 0, 0), Mode.TURN_RIGHT_YAW

    def fast_lap(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        最速ラップ走行モードの制御。

        各フェーズで画像処理やモーター移動距離、ライン検出・青面積判定などの条件に応じて、速度・ターゲット座標・モードを返却し、最速ラップを目指す。
        フェーズ遷移や返却値の詳細は実装内容を参照。
        """

        # ActionChain設計に厳密に合わせる: 初期化はself._initのみ判定し、initialize_actionでフェーズ管理
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase

        # phase0: 領域検出で次フェーズへ。未検出時は中央追従・回避モード返却
        if phase.get_phase() == 0:
            _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow", roi=ROI_COLOR2)
            current_pos = self.get_motor_position(self.course)
            if current_pos < 1000:
                # 右モーター距離が1000未満なら高速で直進し続ける
                return None, (HIGH_SPEED_BASE, HIGH_SPEED_BASE, 0), Mode.AVOID_OBSTACLE

            if yellow_pixel_count > 5000 and current_pos >= 1000:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | yellow_pixel_count={yellow_pixel_count} > 5000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
            else:
                target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                return target_x, (0, 0, HIGH_SPEED_BASE), Mode.AVOID_OBSTACLE

        # phase1: 領域検出で次フェーズへ。未検出時は中心または中央追従・回避モード返却
        if phase.get_phase() == 1:
            yellow_cx, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow", roi=ROI_COLOR2)
            if yellow_pixel_count > 18000:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | yellow_pixel_count={yellow_pixel_count} > 18000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
            else:
                if yellow_cx is not None:
                    target_x = yellow_cx[0]
                else:
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase2: 左旋回（条件成立まで速度調整、到達で次フェーズへ・モーター位置記録）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 350:
                if self.course == "right":
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 350")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # phase3: 直進（条件成立まで定速走行、到達で次フェーズへ・モーター位置記録）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 250:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 250")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.opposite_course))

        # phase4: 条件成立まで定速走行、成立で判定・次フェーズへ（モーター位置記録）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 450:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
                
            if is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3):
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} (horizontal line detected)")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
            else:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE

        # phase5: 右モーター移動距離が所定値未満なら中央追従、以上で次フェーズへ、右モーター位置記録。
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            color_info = self.get_color_sensor_values()
            color_type = color_info["color_type"]
            # 距離450に到達する、もしくはcolor_typeが白以外になったら次フェーズ
            if position_diff >= 450 or color_type != "white":
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 450 or color_type={color_type} != 'white' (color_value={color_info['color']})")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
            else:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.AVOID_OBSTACLE

        # phase6: 左旋回（短距離は低速、長距離は高速、所定値以上で次フェーズへ。所定値未満かつ垂直黒ライン検出で次フェーズへ）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 350:
                vertical_detected = is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=150)
                if vertical_detected:
                    print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} (vertical black line detected)")
                    phase.next_phase()
                    phase.set_position_start("position_start", self.get_motor_position(self.course))
                    return None, None, Mode.AVOID_OBSTACLE
                else:
                    if self.course == "right":
                        return None, (5, 35, 0), Mode.AVOID_OBSTACLE
                    else:
                        return None, (35, 5, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 350")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                return None, None, Mode.AVOID_OBSTACLE

        # phase7: Go to next phase when distance threshold is reached. Otherwise, return get_target_x_by_course with AVOID_OBSTACLE.
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)

            # if is_fast_corner_detected(image, course=self.course):
            #     print(f"[DEBUG] phase7: is_fast_corner_detected=True at current_pos={current_pos}, course={self.course}")

            if position_diff >= 1500:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 1500 (threshold reached)")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase8: After right motor travels threshold, check vertical black line to go to next phase. Otherwise, return get_target_x_by_course with AVOID_OBSTACLE.
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            center_line_detected = is_center_line_detected(image)

            # if is_fast_corner_detected(image, course=self.course):
            #     print(f"[DEBUG] phase8: is_fast_corner_detected=True at current_pos={current_pos}, course={self.course}")

            if position_diff > 2600:
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | position_diff={position_diff} > 2600 (threshold reached)")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
            elif center_line_detected:
                target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                return target_x, (0, 0, HIGH_SPEED_BASE), Mode.AVOID_OBSTACLE
            else:
                # Lock if passed once and now False
                image = fill_green_with_white(image)
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase9: Go to DOUBLE_LOOP if blue area threshold or right motor distance is reached, otherwise continue AVOID_OBSTACLE.
        if phase.get_phase() == 9:
            blue_area = get_blue_line_pixel(image)
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                current_pos = self.get_motor_position(self.course)
                print(f"[DEBUG] mode={Mode.AVOID_OBSTACLE.value} | phase={phase.get_phase()} | blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD} (threshold reached)")
                self.reset_action()
                return target_x, (0, 0, BASE_SPEED), Mode.DOUBLE_LOOP
            else:
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        print("[avoid_obstacle] Unexpected state reached.")
        return None, None, Mode.AVOID_OBSTACLE
