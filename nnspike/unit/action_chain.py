import re
import time  # 時間計測用
from typing import Optional, Tuple  # 型ヒント用

import numpy as np  # 画像処理用
import math  # 角度計算用

# 定数・モード・ROI設定
from nnspike.constants import OFFSET_Y, ROI_CNN, ROI_LINE_TRACING, Mode, BASE_SPEED, HIGH_SPEED_BASE, ROI_LINE_HORIZON3, ROI_LOOP, ROI_LINE_CORNER, ROI_COLOR, ROI_LINE_STRAIGHT, ROI_COLOR2, CAMERA_WIDTH

# --- 閾値定数（全体で統一管理）---
BLUE_AREA_MAX_THRESHOLD = 18000
BLUE_AREA_MIN_THRESHOLD = 3000
FIRST_INTERSECTION_LIMIT = 2000
SECOND_INTERSECTION_LIMIT = 6000
THIRD_INTERSECTION_LIMIT = 8000
FOURTH_INTERSECTION_LIMIT = 10500

YAW_GAIN = 10.0  # ジャイロ補正ゲイン（固定値）

# ロボット本体クラス
from nnspike.unit.etrobot import ETRobot
# 画像処理・ライン/ターゲット検出関数群
from nnspike.utils.control import (
    find_bottle_center,  # ボトル中心検出
    get_line_edges_at_y,  # ライン端抽出
    get_virtual_line_target_x,  # 仮想ライン中心検出
    find_blue_target_center,  # 青ターゲット中心検出
    get_is_blue_line_at_y,  # 青ライン有無判定
    is_x320_on_red_target,  # 画像中央付近で赤ターゲット抽出
    get_red_target_center_x,  # 赤ターゲット中心検出
    is_left_black_line_detected,  # 左黒ライン抽出
    is_lower_horizontal_line_detected,  # 下部水平黒ライン抽出
    is_vertical_black_line_detected,  # 垂直黒ライン抽出
    is_upper_horizontal_line_detected,  # 上部水平黒ライン抽出
    get_blue_line_pixel,  # 青オブジェクト面積抽出
    is_fast_corner_detected,  # コーナー抽出
    fill_green_with_white,    # 緑を白で塗りつぶし
    is_center_line_detected,  # 中央ライン検出関数
    get_offset_pixels,        # 目標位置オフセット計算
)

# 型ヒント用: 要素タプル明示
SpeedTuple = Tuple[int, int]

class PhaseManager:

    def __init__(self, motor_side):
        self._state = {}
        self._state["phase"] = 0
        self._state["position_start"] = None
        self._motor_side = motor_side

    def get_phase(self) -> int:
        phase = self._state.get("phase", 0)
        return phase

    def next_phase(self, current_pos: int, skip: int = 1) -> None:
        self._state["phase"] = self._state.get("phase", 0) + skip
        self.set_position_start("position_start", current_pos)

    def set_position_start(self, key: str, value: int) -> None:
        self._state[key] = value

    def get_position_start(self, key: str) -> int:
        try:
            return int(self._state[key])
        except (KeyError, TypeError, ValueError):
            return 0

    def get_position_diff(self, current_pos) -> int:
        position_start = self.get_position_start("position_start")
        position_diff = abs(current_pos - position_start)
        return position_diff

class ActionChain(object):

    def __init__(self, et: ETRobot, course: str, course_type: str, pid) -> None:
        self.et = et  # ロボット本佁E
        self.course = course  # コース種別
        # コース種別の逆コースを定義
        if course == "right":
            self.opposite_course = "left"
        else:
            self.opposite_course = "right"
        self.course_type = course_type  # 上段/下段コースタイプ（upper/lower）
        self.start_time = 0.0  # アクション開始時刻
        self.current_time = 0.0  # 現在時刻
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN  # 領域定義
        self.center_x = (self.x1 + self.x2) // 2  # 中央X座標
        self.pre_target_x = self.center_x
        self._init = False
        self.pid = pid  # 基本的に外部から渡されたPIDインスタンスのみを使用
        # 加速制御用プライベート変数
        self._acceleration_start_time = None

    def initialize_action(self, motor_side: str = "right"):
        self._phase = PhaseManager(motor_side)
        self._phase.set_position_start("position_start", self.get_motor_position(motor_side))
        self._init = True
        # アクション開始時に加速タイマーをリセット
        self._reset_acceleration_timer()

    def reset_action(self):
        self._init = False
        self._reset_acceleration_timer()

    def _reset_acceleration_timer(self):
        """加速タイマーをリセットする（プライベートメソッド）"""
        self._acceleration_start_time = None

    def _start_acceleration_timer(self):
        """加速タイマーを開始する（プライベートメソッド）"""
        self._acceleration_start_time = time.time()

    def _get_acceleration_elapsed_time(self) -> float:
        """加速開始からの経過時間を取得する（プライベートメソッド）"""
        if self._acceleration_start_time is None:
            return 0.0
        return time.time() - self._acceleration_start_time

    def get_accelerated_base_speed(self, target_speed: int = 30, acceleration_time: float = 1.5) -> int:
        # タイマーが未初期化の場合は自動開始
        if self._acceleration_start_time is None:
            self._start_acceleration_timer()
            
        # 経過時間を計算
        elapsed_time = self._get_acceleration_elapsed_time()
        
        # acceleration_time秒以降は到達速度を確実に返す（無駄な計算回避）
        if elapsed_time >= acceleration_time:
            return target_speed
            
        # acceleration_time秒未満のみ計算実行
        # 2次関数による初期緩やか加速（quadratic ease-in）
        ratio = elapsed_time / acceleration_time
        # 初期は非常に緩やか、その後は一定の加速度で上昇（急激な変化なし）
        quadratic_ratio = ratio * ratio
        
        # 最低速度5から到達速度まで
        min_speed = 5
        calculated_speed = int(min_speed + (target_speed - min_speed) * quadratic_ratio)
        
        return max(min_speed, min(calculated_speed, target_speed))

    def get_motor_position(self, motor_side: str = "right") -> int:
        return self.et.get_motor_relative_position(motor_side)

    def get_color_sensor_values(self) -> dict:
        color_value, color_type = self.et.get_color_sensor()
        return {
            "color": color_value,
            "color_type": color_type
        }

    # --- action_chain用 motor speed計算関数 ---
    def calc_motor_speed(self, target_x: Optional[int], base_speed: int = BASE_SPEED) -> SpeedTuple:
        if base_speed is None or base_speed == 0:
            base_speed = BASE_SPEED
        if target_x is not None:
            offset_pixels = get_offset_pixels(target_x, ROI_CNN)
            theta = math.atan2(offset_pixels, CAMERA_WIDTH)
            steering_correction = self.pid.update(theta)
            left_speed = base_speed - steering_correction
            right_speed = base_speed + steering_correction
        else:
            left_speed = base_speed
            right_speed = base_speed
        left_speed = int(max(0, min(255, left_speed)))
        right_speed = int(max(0, min(255, right_speed)))
        return left_speed, right_speed

    def get_target_x_by_course(self, image, offset_y, course="right") -> int:
        if course == "right":
            _, right_x, _ = get_line_edges_at_y(image, ROI_LINE_TRACING, offset_y, 80)
            target_x = right_x if right_x is not None else self.center_x
        elif course == "left":
            left_x, _, _ = get_line_edges_at_y(image, ROI_LINE_TRACING, offset_y, 80)
            target_x = left_x if left_x is not None else self.center_x
        else:
            target_x = self.center_x
        return int(target_x)
  
    def execute_double_loop(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:

        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.pid.Kp = 50
            self.pid.Ki = 0
            self.pid.Kd = 5
            self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)
            # 開始直後の絶対位置を取得し保持
            self._start_position = self.get_motor_position(self.course)

        phase = self._phase
        current_pos = self.get_motor_position(self.course)
        dist_start = abs(current_pos - self._start_position)

        # phase0: 青領域が条件を満たしたら即次フェーズへ
        if phase.get_phase() == 0:    
            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase0->phase1: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < FIRST_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= FIRST_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase0->phase2: dist_start={dist_start} >= {FIRST_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos, 2)

        # phase1: 青領域が条件未満になったら次フェーズへ
        if phase.get_phase() == 1:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase1->phase2: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < FIRST_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= FIRST_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase1->phase3: dist_start={dist_start} >= {FIRST_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos)

        # phase2: get_blue_line_pixelでBLUE_AREA_THRESHOLD満たしたら即phase3へ
        if phase.get_phase() == 2:
            if dist_start < FIRST_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase2->phase3: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < SECOND_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= SECOND_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase2->phase4: dist_start={dist_start} >= {SECOND_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos, 2)

        # phase3: 青領域が条件未満になったら次フェーズへ
        if phase.get_phase() == 3:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase3->phase4: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < SECOND_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= SECOND_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase3->phase5: dist_start={dist_start} >= {SECOND_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos)

        # phase4: 青領域が条件を満たしたら即次フェーズへ
        if phase.get_phase() == 4:
            if dist_start < SECOND_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase4->phase5: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < THIRD_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= THIRD_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase4->phase6: dist_start={dist_start} >= {THIRD_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos, 2)

        # phase5: 青領域が条件未満になったら直進フェーズへ
        if phase.get_phase() == 5:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase5->phase6: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)  # phase6(直進)へ
            elif dist_start < THIRD_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= THIRD_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase5->phase6: dist_start={dist_start} >= {THIRD_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos)  # phase6へ

        # phase6: 青領域が条件を満たしたら即次フェーズへ
        if phase.get_phase() == 6:
            if dist_start < THIRD_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP

            blue_area = get_blue_line_pixel(image)
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase6->phase7: blue_area={blue_area} > {BLUE_AREA_MAX_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < FOURTH_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= FOURTH_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase6->phase8: dist_start={dist_start} >= {FOURTH_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos, 2)

        # phase7: 青領域が条件未満になったら次フェーズへ
        if phase.get_phase() == 7:
            blue_area = get_blue_line_pixel(image)
            if blue_area < BLUE_AREA_MIN_THRESHOLD:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase7->phase8: blue_area={blue_area} < {BLUE_AREA_MIN_THRESHOLD} | dist_start={dist_start}")
                phase.next_phase(current_pos)
            elif dist_start < FOURTH_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            elif dist_start >= FOURTH_INTERSECTION_LIMIT:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase7->phase8: dist_start={dist_start} >= {FOURTH_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos)

        # phase8: 条件未満ならDOUBLE_LOOP継続、条件到達で次モードへ
        if phase.get_phase() == 8:
            if dist_start < FOURTH_INTERSECTION_LIMIT:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.DOUBLE_LOOP
            else:
                print(f"[DEBUG] mode={Mode.DOUBLE_LOOP.value} | phase8->phase9: dist_start={dist_start} >= {FOURTH_INTERSECTION_LIMIT}")
                phase.next_phase(current_pos)

        # phase9: CARRY_BOTTLE1へ
        if phase.get_phase() == 9:
            self.reset_action()
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
            return (left_speed, right_speed), Mode.CARRY_BOTTLE1

        print("[execute_double_loop] Unexpected state reached.")
        return (0, 0), Mode.DOUBLE_LOOP

    def carry_bottle1_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw_nearest_vertical_pole()
        phase = self._phase
        current_pos = self.get_motor_position(self.course)
        et = self.et

        # 0. 右エッジトレース（赤ピクセル数が一定値を超えたらphase1へ、右モーター初期位置記録）
        if phase.get_phase() == 0:
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            _, _, red_pixel_count = find_bottle_center(image=image, color="red", roi=ROI_COLOR2)
            if red_pixel_count > 3000:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | red_pixel_count={red_pixel_count} > 3000")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            else:
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1

        if phase.get_phase() == 1:
            red_center, _, red_pixel_count = find_bottle_center(image=image, color="red", roi=ROI_COLOR2)
            if red_center is not None and abs(red_center[0] - self.center_x) <= 20:
                self.et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | centered | target_x={red_center[0]} | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            elif red_center is not None:
                if red_center[0] < self.center_x:
                    return (0, 5), Mode.CARRY_BOTTLE1
                else:
                    return (5, 0), Mode.CARRY_BOTTLE1
            else:
                in_tolerance, yaw_error = et.is_start_yaw_error_within(3.0)
                start_yaw = et.get_start_yaw()
                current_yaw = et.get_yaw()
                if in_tolerance:
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | in_tolerance={in_tolerance} | start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    if yaw_error < 0:
                        return (5, 0), Mode.CARRY_BOTTLE1
                    else:
                        return (0, 5), Mode.CARRY_BOTTLE1

        # phase2: 赤ピクセル数 < 500 かつ距離 < 15 で phase3へ。yaw基準セット
        if phase.get_phase() == 2:
            red_center, _, red_pixel_count = find_bottle_center(image=image, color="red", roi=ROI_COLOR2)
            distance = et.get_distance_sensor()
            if (red_pixel_count is not None and red_pixel_count < 500) and (distance > 0 and distance <= 10):
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | red_pixel_count={red_pixel_count} < 500 and distance={distance} > 0 and <= 10 | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                # フェーズ3は停止状態から開始されるため加速タイマーをリセット
                self._reset_acceleration_timer()
                return (0, 0), Mode.CARRY_BOTTLE1
            if red_center is not None:
                target_x = red_center[0]
                accelerated_speed = self.get_accelerated_base_speed()
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=accelerated_speed)
            else:
                accelerated_speed = self.get_accelerated_base_speed()
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed)
            return (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # phase3: 右モーター位置差が閾値（上段1220/下段700）未満なら直進。閾値到達したらphase4へ。yaw基準設定。
        if phase.get_phase() == 3:
            position_diff = phase.get_position_diff(current_pos)
            threshold = 1200 if self.course_type == "upper" else 700
            if position_diff < threshold:
                accelerated_speed = self.get_accelerated_base_speed()
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE1

        # 4. 左旋回（右モーター相対位置差分が一定値未満の間旋回。一定値超えたらphase5へ、右モーター位置記録）
        if phase.get_phase() == 4:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 390:
                if self.course == "right":
                    return (0, 20), Mode.CARRY_BOTTLE1
                else:
                    return (20, 0), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 390")
            phase.next_phase(current_pos)
            # フェーズ5は停止状態から開始されるため加速タイマーをリセット
            self._reset_acceleration_timer()
            return (0, 0), Mode.CARRY_BOTTLE1

        # 5. 直進（右モーターが一定値移動まで。一定値超えたらphase6へ、右モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 5:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 200:
                accelerated_speed = self.get_accelerated_base_speed()
                return (accelerated_speed, accelerated_speed), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 200")
            phase.next_phase(current_pos)
            self.pre_target_x = self.center_x

        # 6. 仮想ライン直進（右モーターが一定値移動まで仮想ライン中心座標取得処理、pre_target_x更新。一定値超えたらphase7へ、右モーター位置記録）
        if phase.get_phase() == 6:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 1500:
                # 仮想ライン中心座標取得処理
                temp_x = get_virtual_line_target_x(image, previous_center_x=self.pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    self.pre_target_x = temp_x
                else:
                    target_x = self.center_x
                    self.pre_target_x = target_x
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 1500")
            phase.next_phase(current_pos)

        # 7. 直進（右モーターが一定値移動まで。一定値超えたらphase8へ、右モーター位置記録）
        if phase.get_phase() == 7:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 1300:
                return (30, 30), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 1300")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE1

        # phase8: 青ターゲット検出または最大回転量到達で次フェーズ。最低回転量以上の旋回処理。
        if phase.get_phase() == 8:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            blue_target_detected = (blue_center is not None and 
                                  abs(blue_center[0] - 320) <= 200)
            position_diff = phase.get_position_diff(current_pos)

            # 最低回転量未満は強制旋回
            if position_diff < 300:
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE1
                else:
                    return (30, 0), Mode.CARRY_BOTTLE1

            # 最低回転量以上になったら判定開始
            if (not blue_target_detected) and (position_diff < 500):
                if self.course == "right":
                    return (0, 20), Mode.CARRY_BOTTLE1
                else:
                    return (20, 0), Mode.CARRY_BOTTLE1

            et.set_start_yaw()
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_target_detected={blue_target_detected} or position_diff={position_diff} >= 500 | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f} | blue_pixel_count={blue_pixel_count} | blue_center={blue_center}")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE1

        # phase9: 青ターゲットを中央に合わせる。中央付近なら即停止、そうでなければ回転のみのシンプルロジック。
        if phase.get_phase() == 9:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            # blue_centerは(center_x, top_y)のタプル、top_yは必ず保持
            if blue_center is not None:
                blue_center_x, blue_top_y = blue_center[0], blue_center[1]
            else:
                blue_center_x, blue_top_y = None, None
            
            if blue_center is not None and abs(blue_center[0] - self.center_x) <= 20:
                et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | centered | target_x={blue_center[0]} | center_x={self.center_x} | blue_top_y={blue_top_y} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            elif blue_center is not None:
                if blue_center[0] < self.center_x:
                    return (0, 5), Mode.CARRY_BOTTLE1
                else:
                    return (5, 0), Mode.CARRY_BOTTLE1
            else:
                in_tolerance, yaw_error = et.is_start_yaw_error_within(3.0)
                start_yaw = et.get_start_yaw()
                current_yaw = et.get_yaw()
                if in_tolerance:
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | in_tolerance={in_tolerance} | start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    if yaw_error < 0:
                        return (5, 0), Mode.CARRY_BOTTLE1
                    else:
                        return (0, 5), Mode.CARRY_BOTTLE1

        # phase10: 青ピクセル数>1000でcenter追従、<=300で次フェーズ、それ以外はyaw維持直進
        if phase.get_phase() == 10:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_pixel_count > 1000:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} > 1000 | blue_center={blue_center} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                if blue_center is not None:
                    et.set_start_yaw()
                    left_speed, right_speed = self.calc_motor_speed(blue_center[0], base_speed=20)
                else:
                    left_speed, right_speed = et.yaw_straight_control(base_speed=20, adjust_speed=2, deadband=2)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1
            elif blue_pixel_count <= 300:
                et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} <= 300 | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
            else:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} | blue_center={blue_center} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                left_speed, right_speed = et.yaw_straight_control(base_speed=20, adjust_speed=2, deadband=2)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # phase11: 色センサーが青検出でphase14へ移行（停止）。それ以外はヨー維持で直進（低速）。
        if phase.get_phase() == 11:
            position_diff = phase.get_position_diff(current_pos)
            threshold = 500
            color_info = self.get_color_sensor_values()
            color_type = color_info["color_type"]
            
            if color_type == "blue" or position_diff >= threshold:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold} or color_type={color_type} (color_value={color_info['color']}) | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            else:
                left_speed, right_speed = et.yaw_straight_control(base_speed=10, deadband=2)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # phase12: 状態リセットしBACK_AND_TURN1へ遷移。
        if phase.get_phase() == 12:
            self.reset_action()
            return (0, 0), Mode.BACK_AND_TURN1

        print("[carry_bottle1_relative] Unexpected state reached.")
        return (0, 0), Mode.CARRY_BOTTLE1

    def back_and_turn1_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        current_pos = self.get_motor_position(self.course)

        # 0. 後退（コース側モーターが所定値移動まで。所定値超えたらphase1へ、モーター位置記録）
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 600:
                return (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN1
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 600")
            phase.next_phase(current_pos)

        # 1. 左旋回（最低回転量は必ず旋回。最低回転量超えてからターゲット検出または最大回転量到達まで旋回。条件満たせばphase2へ）
        if phase.get_phase() == 1:
            red_target_detected = is_x320_on_red_target(image, x_tolerance=80)
            position_diff = phase.get_position_diff(current_pos)
            # 最低回転量は必ず旋回
            if position_diff < 450:
                if self.course == "right":
                    return (0, 30), Mode.BACK_AND_TURN1
                else:
                    return (30, 0), Mode.BACK_AND_TURN1
            # 最低回転量超えてから、ターゲット検出または最大回転量到達まで継続
            if (not red_target_detected) and (position_diff < 940):
                if self.course == "right":
                    return (0, 30), Mode.BACK_AND_TURN1
                else:
                    return (30, 0), Mode.BACK_AND_TURN1
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 940 or red_target_detected={red_target_detected}")
            phase.next_phase(current_pos)
            return (0, 0), Mode.BACK_AND_TURN1

        # 2. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        if phase.get_phase() == 2:
            self.reset_action()
            return (0, 0), Mode.CARRY_BOTTLE2

        print("[back_and_turn1_relative] Unexpected state reached.")
        return (0, 0), Mode.BACK_AND_TURN1

    def carry_bottle2_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        current_pos = self.get_motor_position(self.course)

        # 0. 赤ターゲット中心追従（青ピクセル数が閾値未満の間は赤中心追従、閾値以上で次フェーズへ）
        if phase.get_phase() == 0:
            blue_center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
            if blue_pixel_count < 18000:
                # 赤ターゲット中心追従
                red_center_x = get_red_target_center_x(image)
                # 赤センター最優先
                if red_center_x is not None:
                    target_x = red_center_x
                elif blue_pixel_count >= 5000 and blue_center is not None:
                    target_x = blue_center[0]
                else:
                    target_x = self.center_x
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} >= 18000")
                phase.next_phase(current_pos)

        # 1. 青ボトル中心追従（青ピクセル数が十分な間はcenter追従、少なくなったらphase2へ移行し右モーター位置記録）
        if phase.get_phase() == 1:
            blue_center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
            target_x = blue_center[0] if blue_center is not None else self.center_x
            # 上限距離チェック
            position_diff = phase.get_position_diff(current_pos)
            if position_diff >= 1000 or blue_pixel_count < 5000:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 1000 or blue_pixel_count={blue_pixel_count} < 5000")
                phase.next_phase(current_pos)
            else:
                # 青ボトル中心x座標へ追従
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 2. 右モーターが所定値移動までcenter追従。所定値超えたら次フェーズへ、右モーター位置記録
        if phase.get_phase() == 2:
            blue_center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 200:
                if blue_center is not None:
                    target_x = blue_center[0]
                else:
                    target_x = self.center_x
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 200")
            phase.next_phase(current_pos)

        # 3. 左黒ライン検出まで左旋回。最大回転量まで。検出または最大回転量到達で次フェーズへ、右モーター位置記録
        if phase.get_phase() == 3:
            min_limit = 700
            max_limit = 1200
            position_diff = phase.get_position_diff(current_pos)
            # 最小閾値未満は検知開始しない
            if position_diff < min_limit and position_diff < max_limit:
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE2
                else:
                    return (30, 0), Mode.CARRY_BOTTLE2

            # 最小閾値以上になったら判定開始
            line_detected = is_left_black_line_detected(image, self.course)

            if (not line_detected) and (position_diff < max_limit):
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE2
                else:
                    return (30, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | line_detected={line_detected} | position_diff={position_diff} >= {max_limit}")
            phase.next_phase(current_pos)

        # 4. 直進（コース側モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase5へ、モーター位置記録）
        if phase.get_phase() == 4:
            threshold = 870 if self.course_type == "upper" else 1300
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < threshold:
                return (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold}")
            phase.next_phase(current_pos)

        # 5. 左旋回（コース側モーターが所定値移動まで、courseに応じて旋回方向決定。所定値超えたらphase6へ、モーター位置記録）
        if phase.get_phase() == 5:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 350:
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE2
                else:
                    return (30, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 350")
            phase.next_phase(current_pos)

        # 6. 直進（コース側モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase7へ、モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 6:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 100:
                return (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 100")
            phase.next_phase(current_pos)
            self.pre_target_x = self.center_x

        # 7. 仮想ライン直進（コース側モーターが所定値移動まで仮想ライン中心座標取得処理、pre_target_x更新。所定値超えたらphase8へ、モーター位置記録）
        if phase.get_phase() == 7:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 1400:
                # 仮想ライン中心座標取得処理
                temp_x = get_virtual_line_target_x(image, previous_center_x=self.pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    self.pre_target_x = temp_x
                else:
                    target_x = self.center_x
                    self.pre_target_x = target_x
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 1400")
            phase.next_phase(current_pos)

        # 8. 直進（コース側モーターが所定値移動まで、両輪BASE_SPEED。所定値超えたらphase9へ、モーター位置記録）
        if phase.get_phase() == 8:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 400:
                return (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 400")
            phase.next_phase(current_pos)

        # 9. 左旋回（青ターゲット検出まで、最低回転量・最大回転量。条件満たせばphase10へ、右モーター位置記録）
        if phase.get_phase() == 9:
            blue_center, _, _ = find_blue_target_center(image)
            blue_target_detected = (blue_center is not None and 
                                  abs(blue_center[0] - 320) <= 60)
            position_diff = phase.get_position_diff(current_pos)
            # 最低回転量は必ず旋回
            if position_diff < 300:
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE2
                else:
                    return (30, 0), Mode.CARRY_BOTTLE2
            # 最低回転量超えてから、ターゲット検出または最大回転量到達まで継続
            if (not blue_target_detected) and (position_diff < 500):
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE2
                else:
                    return (30, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_target_detected={blue_target_detected} or position_diff={position_diff} >= 500")
            phase.next_phase(current_pos)

        # 10. 青検出（青ピクセル数が一定値を超えたらphase11へ、最大回転量。条件満たせば右モーター位置記録）
        if phase.get_phase() == 10:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_center is not None:
                target_x = blue_center[0]
            else:
                target_x = self.center_x
            position_diff = phase.get_position_diff(current_pos)
            if blue_pixel_count > 1000 or position_diff >= 400:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} > 1000 or position_diff={position_diff} >= 400")
                phase.next_phase(current_pos)
            else:
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 11. 青ピクセルが一定値以下まで減るまでcenter追従（一定値以下でphase12へ、最大回転量。条件満たせば右モーター位置記録）
        if phase.get_phase() == 11:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_center is not None:
                target_x = blue_center[0]
            else:
                target_x = self.center_x
            threshold = 400 if self.course_type == "upper" else 1000
            position_diff = phase.get_position_diff(current_pos)
            if blue_pixel_count <= 300 or position_diff >= threshold:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} <= 300 or position_diff={position_diff} >= {threshold}")
                phase.next_phase(current_pos)
            else:
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 12. コース側モーターが所定値移動までcenter追従。所定値超えたらphase13へ
        if phase.get_phase() == 12:
            position_diff = phase.get_position_diff(current_pos)
            color_info = self.get_color_sensor_values()
            color_type = color_info["color_type"]
            if position_diff >= 300 or color_type == "other":
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 300 or color_type={color_type} (color_value={color_info['color']})")
                phase.next_phase(current_pos)
            else:
                blue_center, _, _ = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
                if blue_center is not None:
                    target_x = blue_center[0]
                else:
                    target_x = self.center_x
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 13. 状態リセットしBACK_AND_TURN2へ遷移
        if phase.get_phase() == 13:
            self.reset_action()
            return (0, 0), Mode.BACK_AND_TURN2

        print("[carry_bottle2_relative] Unexpected state reached.")
        return (0, 0), Mode.CARRY_BOTTLE2

    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.opposite_course)
        phase = self._phase
        current_pos = self.get_motor_position(self.opposite_course)

        # 0. 左モーターが抽象的な基準位置まで後退（両輪定速）。条件成立で次フェーズへ、モーター位置記録。
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 570:
                return (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN2
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 570")
            phase.next_phase(current_pos)

        # 1. 右旋回（抽象的な基準位置までは必ず旋回。条件成立後、ライン検出または基準位置到達まで所定速度で継続。条件成立で次フェーズへ）
        if phase.get_phase() == 1:
            position_diff = phase.get_position_diff(current_pos)
            horizontal_line_detected = is_upper_horizontal_line_detected(image)
            # 抽象的な基準位置までは必ず旋回
            if position_diff < 200:
                if self.course == "right":
                    return (30, 0), Mode.BACK_AND_TURN2
                else:
                    return (0, 30), Mode.BACK_AND_TURN2
            # 基準位置到達後、ライン検出または別基準位置到達まで継続
            if (not horizontal_line_detected) and (position_diff < 500):
                if self.course == "right":
                    return (30, 0), Mode.BACK_AND_TURN2
                else:
                    return (0, 30), Mode.BACK_AND_TURN2
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN2.value} | phase={phase.get_phase()} | horizontal_line_detected={horizontal_line_detected} | position_diff={position_diff} >= 500")
            # 条件を満たしたので次のフェーズへ
            phase.next_phase(current_pos)

        # 2. 終了: 状態リセットし目標モードへ遷移（抽象化）
        if phase.get_phase() == 2:
            self.reset_action()
            return (0, 0), Mode.HEAD_GOAL

        print("[back_and_turn2_relative] Unexpected state reached.")
        return (0, 0), Mode.BACK_AND_TURN2

    def heading_goal_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        current_pos = self.get_motor_position(self.course)

        # 0. 所定のintersection_yで黒水平ライン検出まで中央追従（距離制限なし）。検出でphase1へ、右モーター位置記録。
        if phase.get_phase() == 0:
            if is_lower_horizontal_line_detected(image, intersection_y=450):
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | horizontal_line_detected")
                phase.next_phase(current_pos)
            else:
                target_x = self.center_x
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.HEAD_GOAL

        # 1. コース側モーターの移動距離が所定値未満なら中央追従、所定値以上で次フェーズへ遷移。到達でモーター位置記録。
        if phase.get_phase() == 1:
            position_diff = phase.get_position_diff(current_pos)
            color_info = self.get_color_sensor_values()
            color_type = color_info["color_type"]
            if position_diff >= 350 or color_type != "white":
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 350 or color_type={color_type} (color_value={color_info['color']})")
                phase.next_phase(current_pos)
            else:
                return (BASE_SPEED, BASE_SPEED), Mode.HEAD_GOAL

        # 2. 左旋回（courseに応じて左旋回。垂直黒ライン検出または移動距離上限到達でphase3へ）
        if phase.get_phase() == 2:
            position_diff = phase.get_position_diff(current_pos)
            vertical_line_detected = is_vertical_black_line_detected(image)
            # Continue turning left. If vertical black line detected or position limit reached, go to phase3
            if (not vertical_line_detected) and (position_diff < 500):
                if self.course == "right":
                    return (0, 30), Mode.HEAD_GOAL
                else:
                    return (30, 0), Mode.HEAD_GOAL
            print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | vertical_line_detected={vertical_line_detected} position_diff={position_diff} >= 500")
            phase.next_phase(current_pos)

        # 3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央。青ライン検出時に右モーター位置記録）
        if phase.get_phase() == 3:
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | blue_line={blue_line}")
                phase.next_phase(current_pos)
            else:
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.HEAD_GOAL

        # 4. 青ライン検出後、コース側モーターの移動距離が所定値未満の間は左エッジトレース、到達でPAUSE（状態リセット）
        if phase.get_phase() == 4:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff >= 300:
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | position_diff={position_diff} | current_pos={current_pos} >= 300")
                phase.next_phase(current_pos)
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x)
                return (left_speed, right_speed), Mode.HEAD_GOAL

        # 5. 青ライン検出後、コース側モーターの移動距離が所定値未満の間は直進、到達で次のフェーズへ
        if phase.get_phase() == 5:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff >= 200:
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | position_diff={position_diff} | current_pos={current_pos} >= 200")
                phase.next_phase(current_pos)
            else:
                # go straight
                return (BASE_SPEED, BASE_SPEED), Mode.HEAD_GOAL

        # 6. 所定距離到達で状態リセットしPAUSE
        if phase.get_phase() == 6:
            self.reset_action()
            return (0, 0), Mode.PAUSE

        print("[heading_goal_relative] Unexpected state reached.")
        return (0, 0), Mode.HEAD_GOAL

    def eye_blue(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        # 初回呼び出し時のみ初期化
        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw_nearest_vertical_pole()
            self._phase2_timer = None
        et = self.et
        phase = self._phase
        current_pos = self.get_motor_position(self.course)

        # phase0: 右モーター位置差500未満なら直進。超えたらphase1へ。右モーター位置記録
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 500:
                left_speed, right_speed = et.yaw_straight_control(base_speed=BASE_SPEED, adjust_speed=2)
                return (left_speed, right_speed), Mode.EYE_BLUE
            else:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 500 | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)

        # phase1: 青ターゲット検出 or 最低回転量後に検出 or 最大回転量到達で次フェーズへ。下段コースは2フェーズスキップ
        if phase.get_phase() == 1:
            blue_center, _, _ = find_blue_target_center(image)
            blue_target_detected = (blue_center is not None and 
                                  abs(blue_center[0] - 320) <= 200)
            position_diff = phase.get_position_diff(current_pos)
            max_limit = 500
            position_limit_reached = position_diff >= max_limit
            if ((position_diff >= 300 and blue_target_detected) or position_limit_reached):
                # 現在のヨー角でスタートヨーを設定
                self.et.set_start_yaw()
                blue_center, _, blue_pixel_count = find_blue_target_center(image)
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {max_limit} or (position_diff={position_diff} >= 300 and blue_target_detected={blue_target_detected}) | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f} | blue_pixel_count={blue_pixel_count} | blue_center={blue_center}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE
            else:
                if self.course == "right":
                    return (0, 20), Mode.EYE_BLUE
                else:
                    return (20, 0), Mode.EYE_BLUE

        # phase2: 青ターゲット中心合わせ。中央付近なら即停止、そうでなければ回転のみのシンプルロジック
        if phase.get_phase() == 2:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | blue_center={blue_center} | blue_pixel_count={blue_pixel_count}")
            if blue_center is not None and abs(blue_center[0] - self.center_x) <= 20:
                self.et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | centered | target_x={blue_center[0]} | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE
            elif blue_center is not None:
                if blue_center[0] < self.center_x:
                    print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | rotate left | blue_center[0]={blue_center[0]} | blue_pixel_count={blue_pixel_count}")
                    return (0, 5), Mode.EYE_BLUE
                else:
                    print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | rotate right | blue_center[0]={blue_center[0]} | blue_pixel_count={blue_pixel_count}")
                    return (5, 0), Mode.EYE_BLUE
            else:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | blue_center is None | blue_pixel_count={blue_pixel_count}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE
                
        # phase3: 青ピクセル数>1000でcenter追従、<=300で次フェーズ、それ以外はyaw維持直進
        if phase.get_phase() == 3:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_pixel_count > 1000:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} > 1000 | blue_center={blue_center} | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                if blue_center is not None:
                    self.et.set_start_yaw()
                    left_speed, right_speed = self.calc_motor_speed(blue_center[0], base_speed=20)
                else:
                    left_speed, right_speed = et.yaw_straight_control(base_speed=20, adjust_speed=2, deadband=2)
                return (left_speed, right_speed), Mode.EYE_BLUE
            elif blue_pixel_count <= 300:
                self.et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} <= 300 | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                phase.next_phase(current_pos)
            else:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} | blue_center={blue_center} | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                left_speed, right_speed = et.yaw_straight_control(base_speed=20, adjust_speed=2, deadband=2)
                return (left_speed, right_speed), Mode.EYE_BLUE

        # phase4: 色センサーが青検出でphase5へ停止。それ以外はヨー維持で直進（低速）
        if phase.get_phase() == 4:
            threshold = 1000
            color_info = self.get_color_sensor_values()
            color_type = color_info["color_type"]
            position_diff = phase.get_position_diff(current_pos)
            if color_type == "blue" or position_diff >= threshold:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold} or color_type={color_type} (color_value={color_info['color']}) | set_start_yaw={self.et.get_start_yaw():.2f} | current_yaw={self.et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE
            else:
                # ここでfind_blue_target_centerは不要。ヨー維持で直進（低速）
                left_speed, right_speed = et.yaw_straight_control(base_speed=10, deadband=2)
                return (left_speed, right_speed), Mode.EYE_BLUE

        # phase5: 状態リセットしPAUSEへ遷移
        if phase.get_phase() == 5:
            self.reset_action()
            return (0, 0), Mode.PAUSE

        print("[eye_blue] Unexpected state reached.")
        return (0, 0), Mode.EYE_BLUE

