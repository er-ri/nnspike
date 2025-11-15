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
    calc_blue_target_distance,  # 青ターゲット距離計算
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

    def __init__(self, motor_side: str) -> None:
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

    def get_position_diff(self, current_pos: int) -> int:
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
        # 距離計算変数 - 絶対にNoneにならない
        self._calculated_distance: int = 300  # デフォルト値で必ず初期化

    def initialize_action(self, motor_side: str = "right") -> None:
        self._phase = PhaseManager(motor_side)
        self._phase.set_position_start("position_start", self.get_motor_position(motor_side))
        self._init = True
        # アクション開始時に加速タイマーをリセット
        self._reset_acceleration_timer()

    def reset_action(self) -> None:
        self._init = False
        self._reset_acceleration_timer()

    def _reset_acceleration_timer(self) -> None:
        """加速タイマーをリセットする（プライベートメソッド）"""
        self._acceleration_start_time = None

    def _start_acceleration_timer(self) -> None:
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
            steering_correction = self.pid.update(theta, base_speed)
            left_speed = base_speed - steering_correction
            right_speed = base_speed + steering_correction
        else:
            left_speed = base_speed
            right_speed = base_speed
        left_speed = int(max(0, min(255, left_speed)))
        right_speed = int(max(0, min(255, right_speed)))
        return left_speed, right_speed

    def get_target_x_by_course(self, image: np.ndarray, offset_y: int, course: str = "right") -> int:
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
                # この分岐だけ画像を緑→白で前処理してからターゲットを取得する
                proc_img = fill_green_with_white(image.copy())
                target_x = self.get_target_x_by_course(proc_img, OFFSET_Y, self.course)
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
            # プライベート変数の初期化
            self._calculated_distance = 300  # デフォルト値
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
            if red_center is not None and abs(red_center[0] - self.center_x) <= 10:
                et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | centered | target_x={red_center[0]} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            elif red_center is not None:
                if red_center[0] < self.center_x:
                    return (0, 5), Mode.CARRY_BOTTLE1
                else:
                    return (5, 0), Mode.CARRY_BOTTLE1
            else:
                in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
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

        # phase2: 赤ピクセル数 < 500 かつ距離 > 0 かつ距離 <= 10 で phase3へ。yaw基準セット
        if phase.get_phase() == 2:
            red_center, _, red_pixel_count = find_bottle_center(image=image, color="red", roi=ROI_COLOR2)
            distance = et.get_distance_sensor()
            if (red_pixel_count is not None and red_pixel_count < 500) and (distance > 0 and distance <= 10):
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | red_pixel_count={red_pixel_count} < 500 and distance={distance} > 0 and <= 10 | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
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

        # phase3: 右モーター位置差がコース種別ごとの閾値未満なら直進。閾値到達したらphase4へ。yaw基準設定。
        if phase.get_phase() == 3:
            position_diff = phase.get_position_diff(current_pos)
            threshold = 1170 if self.course_type == "upper" else 700
            if position_diff < threshold:
                accelerated_speed = self.get_accelerated_base_speed()
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE1
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
            phase.next_phase(current_pos)
            self._reset_acceleration_timer()
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
            self._reset_acceleration_timer()
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

        # 7. 直進(右モーターが一定値移動まで。一定値超えたらphase8へ、右モーター位置記録)
        if phase.get_phase() == 7:
            position_diff = phase.get_position_diff(current_pos)
            threshold = 1300 if self.course_type == "upper" else 1900
            if position_diff < threshold:
                # upper区間(1300まで): 左30/右30均等
                # lower区間(1300~1900): 右コースなら左35/右30で右寄せ、左コースなら左30/右35で左寄せ
                if self.course_type == "upper" or position_diff < 1300:
                    return (30, 30), Mode.CARRY_BOTTLE1
                elif self.course == "right":
                    return (32, 30), Mode.CARRY_BOTTLE1
                else:
                    return (30, 32), Mode.CARRY_BOTTLE1
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold}")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE1

        # phase8: 青ターゲット検出または最大回転量到達で次フェーズ。最低回転量以上の旋回処理。
        if phase.get_phase() == 8:
            position_diff = phase.get_position_diff(current_pos)
            # 最低回転量未満は強制旋回（青ターゲット検出しない）
            if position_diff < 300:
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE1
                else:
                    return (30, 0), Mode.CARRY_BOTTLE1

            # 300以上で青ターゲット検出・判定
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_center={blue_center} | blue_pixel_count={blue_pixel_count}")
            blue_target_detected = (blue_center is not None and abs(blue_center[0] - 320) <= 200)

            # 青ターゲット未検出かつ700未満なら旋回継続
            if (not blue_target_detected) and (position_diff < 700):
                if self.course == "right":
                    return (0, 20), Mode.CARRY_BOTTLE1
                else:
                    return (20, 0), Mode.CARRY_BOTTLE1

            # 青ターゲット検出または700以上で次フェーズへ
            et.set_start_yaw()
            # 青ターゲット検出時は距離計算して保存（フェーズ10で使用）
            if blue_target_detected and blue_center is not None:
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | calc_blue_target_distance: X={blue_center[0]}, Y={blue_center[1]}, distance={calculated_distance}, pixels={blue_pixel_count}")
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_target_detected={blue_target_detected} or position_diff={position_diff} >= 700 | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f} | blue_pixel_count={blue_pixel_count} | blue_center={blue_center} | _calculated_distance={self._calculated_distance}")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE1

        # phase9: 青ターゲットを中心に合わせるだけ
        if phase.get_phase() == 9:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            
            if blue_center is not None and blue_center[1] > 10:
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | calc_blue_target_distance: X={blue_center[0]}, Y={blue_center[1]}, distance={calculated_distance}, pixels={blue_pixel_count}")
                if abs(blue_center[0] - self.center_x) <= 10:
                    et.set_start_yaw()
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | centered | blue_center=({blue_center[0]}, {blue_center[1]}) | pixels={blue_pixel_count} | _calculated_distance={self._calculated_distance} | proceed to phase10")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    if blue_center[0] < self.center_x:
                        return (0, 5), Mode.CARRY_BOTTLE1
                    else:
                        return (5, 0), Mode.CARRY_BOTTLE1
            else:
                in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
                if in_tolerance:
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | no_blue_target | yaw_ok | proceed to phase10")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    if yaw_error < 0:
                        return (5, 0), Mode.CARRY_BOTTLE1
                    else:
                        return (0, 5), Mode.CARRY_BOTTLE1

        # phase10: 青ターゲットのy座標が300になるまでゆっくり直進
        if phase.get_phase() == 10:
            # 標準的な距離計算を使用
            distance_from_start = phase.get_position_diff(current_pos)
            # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} | _calculated_distance={self._calculated_distance}")
            # 距離制限チェック - 計算距離に到達した場合は次フェーズへ
            if distance_from_start >= self._calculated_distance:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} >= _calculated_distance={self._calculated_distance} | proceed to tracking phase")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_center is not None and blue_center[1] > 10:
                calculated_distance = calc_blue_target_distance(blue_center)
                # --- print出し方統一 ---
                # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_center=({blue_center[0]}, {blue_center[1]}) | blue_pixel_count={blue_pixel_count} | calc_blue_target_distance={calculated_distance} | _calculated_distance={self._calculated_distance}")
                if blue_center[1] >= 300:
                    if calculated_distance is not None:
                        self._calculated_distance = calculated_distance
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | blue_y={blue_center[1]} >= 300 | proceed to tracking phase | _calculated_distance={self._calculated_distance} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    target_x = blue_center[0]
                    accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                    left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=accelerated_speed)
                    return (left_speed, right_speed), Mode.CARRY_BOTTLE1
            else:
                # --- print出し方統一 ---
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | no_blue_target | _calculated_distance={self._calculated_distance} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                et.set_start_yaw()
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1

        # phase11: もう一度青ターゲットを中心に合わせる（EYE_BLUEのフェーズ2と同じ処理）
        if phase.get_phase() == 11:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_center is not None and blue_center[1] > 10:
                # 距離計算して保存（フェーズ12で使用）
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    # print(f"[calc_blue_target_distance] X={blue_center[0]}, Y={blue_center[1]} → distance={calculated_distance} | pixels={blue_pixel_count}")
                # 中心に合わせる判定（±5ピクセル以内）
                if abs(blue_center[0] - self.center_x) <= 5:
                    # 中心に合った→次フェーズへ
                    et.set_start_yaw()
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | centered | blue_center=({blue_center[0]}, {blue_center[1]}) | pixels={blue_pixel_count} | _calculated_distance={self._calculated_distance} | proceed to phase12")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    # 中心に向けて旋回
                    if blue_center[0] < self.center_x:
                        return (0, 5), Mode.CARRY_BOTTLE1  # 左旋回
                    else:
                        return (5, 0), Mode.CARRY_BOTTLE1  # 右旋回
            else:
                # 青ターゲットが見つからない→ヨー角調整
                in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
                if in_tolerance:
                    # ヨー角OK→フェーズ12へ
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | no_blue_target | yaw_ok | proceed to phase12")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE1
                else:
                    # ヨー角調整
                    if yaw_error < 0:
                        return (5, 0), Mode.CARRY_BOTTLE1
                    else:
                        return (0, 5), Mode.CARRY_BOTTLE1

        # phase12: 青ターゲット追跡または計算距離まで直進。距離到達または青検出で次フェーズへ。
        if phase.get_phase() == 12:
            # 標準的な距離計算を使用
            distance_from_start = phase.get_position_diff(current_pos)
            # 計算距離到達で停止
            if distance_from_start >= (self._calculated_distance + 10):

                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} >= _calculated_distance={self._calculated_distance} | STOP | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE1
            # 青ターゲットの中心追従は行わず、常にyaw_straight_controlで直進
            accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
            left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
            start_yaw = et.get_start_yaw()
            current_yaw = et.get_yaw()
            yaw_error = current_yaw - start_yaw
            # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE1.value} | phase={phase.get_phase()} | start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
            return (left_speed, right_speed), Mode.CARRY_BOTTLE1

        # phase13: 状態リセットしBACK_AND_TURN1へ遷移。
        if phase.get_phase() == 13:
            self.reset_action()
            return (0, 0), Mode.BACK_AND_TURN1

        print("[carry_bottle1_relative] Unexpected state reached.")
        return (0, 0), Mode.CARRY_BOTTLE1

    def back_and_turn1_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
            self._wait_start_time = None
        phase = self._phase
        current_pos = self.get_motor_position(self.course)

        # 0. 後退（コース側モーターが所定値移動まで。所定値超えたらphase1へ、モーター位置記録）
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 600:
                return (-30, -30), Mode.BACK_AND_TURN1
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 600")
            phase.next_phase(current_pos)
            self._wait_start_time = None

        # 1. 2秒待機フェーズ
        if phase.get_phase() == 1:
            if self._wait_start_time is None:
                self._wait_start_time = time.time()
            elapsed = time.time() - self._wait_start_time
            if elapsed >= 2.0:
                phase.next_phase(current_pos)
                self._wait_start_time = None
                return (0, 0), Mode.BACK_AND_TURN1
            # 待機中は停止
            return (0, 0), Mode.BACK_AND_TURN1

        # 2. 左旋回（最低回転量は必ず旋回。最低回転量超えてからターゲット検出または最大回転量到達まで旋回。条件満たせばphase3へ）
        if phase.get_phase() == 2:
            position_diff = phase.get_position_diff(current_pos)
            if self.course_type != "upper":
                limit = 400
                if position_diff >= limit:
                    print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {limit} (immediate next phase)")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.BACK_AND_TURN1
                else:
                    # 350未満は常に30で旋回
                    if self.course == "right":
                        return (0, 30), Mode.BACK_AND_TURN1
                    else:
                        return (30, 0), Mode.BACK_AND_TURN1
            # upperのときは既存ロジックを一切変更しない
            threshold = 450
            limit = 700
            if position_diff < threshold:
                if self.course == "right":
                    return (0, 30), Mode.BACK_AND_TURN1
                else:
                    return (30, 0), Mode.BACK_AND_TURN1
            red_target_detected = is_x320_on_red_target(image, x_tolerance=80)
            if (not red_target_detected) and (position_diff < limit):
                if self.course == "right":
                    return (0, 20), Mode.BACK_AND_TURN1
                else:
                    return (20, 0), Mode.BACK_AND_TURN1
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {limit} or red_target_detected={red_target_detected}")
            phase.next_phase(current_pos, skip=2)
            return (0, 0), Mode.BACK_AND_TURN1

        # 青ボトル検知・追従・遷移判定
        if phase.get_phase() == 3:
            position_diff = phase.get_position_diff(current_pos)
            threshold = 300
            max_threshold = 600
            if position_diff < threshold:
                return (30, 30), Mode.BACK_AND_TURN1
            elif position_diff < max_threshold:
                blue_center, _, blue_pixel_count = find_bottle_center(image=image, color="blue", roi=ROI_COLOR)
                if blue_pixel_count is None or blue_pixel_count < 500:
                    return (30, 30), Mode.BACK_AND_TURN1
                elif blue_pixel_count >= 3000:
                    phase.next_phase(current_pos)
                    print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | blue_pixel_count={blue_pixel_count} >= 3000 | position_diff={position_diff} >= {threshold} and < {max_threshold}")
                else:
                    target_x = blue_center[0] if blue_center is not None else self.center_x
                    left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                    return (left_speed, right_speed), Mode.BACK_AND_TURN1
            else:
                print(f"[DEBUG] mode={Mode.BACK_AND_TURN1.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {max_threshold} | next phase (go straight)")
                phase.next_phase(current_pos)

        # 4. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        if phase.get_phase() == 4:
            self.reset_action()
            return (0, 0), Mode.CARRY_BOTTLE2

        print("[back_and_turn1_relative] Unexpected state reached.")
        return (0, 0), Mode.BACK_AND_TURN1

    def carry_bottle2_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
            # プライベート変数の初期化
            # デフォルト値。course_type が "upper" 以外のときは上限を 800 に引き上げる
            self._calculated_distance = 800 if self.course_type != "upper" else 300
        phase = self._phase
        current_pos = self.get_motor_position(self.course)
        et = self.et

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
                    return (0, 20), Mode.CARRY_BOTTLE2
                else:
                    return (20, 0), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | line_detected={line_detected} | position_diff={position_diff} >= {max_limit}")
            et.set_start_yaw_nearest_vertical_pole()
            phase.next_phase(current_pos)

        # 4. 直進（コース側モーターが所定値移動まで、加速度付き直進。所定値超えたらphase5へ、モーター位置記録）
        if phase.get_phase() == 4:
            threshold = 820 if self.course_type == "upper" else 1300
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < threshold:
                accelerated_speed = self.get_accelerated_base_speed()
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold}")
            phase.next_phase(current_pos)
            self._reset_acceleration_timer()
            return (0, 0), Mode.CARRY_BOTTLE2

        # 5. 左旋回（コース側モーターが所定値移動まで、courseに応じて旋回方向決定。所定値超えたらphase6へ、モーター位置記録）
        if phase.get_phase() == 5:
            position_diff = phase.get_position_diff(current_pos)
            # upper以外は390まで曲がれ
            limit = 350 if self.course_type == "upper" else 390
            if position_diff < limit:
                if self.course == "right":
                    return (0, 20), Mode.CARRY_BOTTLE2
                else:
                    return (20, 0), Mode.CARRY_BOTTLE2
            # 一定値超えたら次フェーズへ
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 350")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE2

        # 6. 直進（コース側モーターが所定値移動まで、加速度付き直進。所定値超えたらphase7へ、モーター位置記録、pre_target_x初期化）
        if phase.get_phase() == 6:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 100:
                accelerated_speed = self.get_accelerated_base_speed()
                return (accelerated_speed, accelerated_speed), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 100")
            phase.next_phase(current_pos)
            self._reset_acceleration_timer()
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
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 1400")
            phase.next_phase(current_pos)

        # 8. 直進（コース側モーターが所定値移動まで、両輪30。所定値超えたらphase9へ、モーター位置記録）
        if phase.get_phase() == 8:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 400:
                return (30, 30), Mode.CARRY_BOTTLE2
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 400")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE2

        # phase9: 青ターゲット検出または最大回転量到達で次フェーズ。最低回転量以上の旋回処理。
        # carry_bottle1のphase8に相当
        if phase.get_phase() == 9:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 350:
                if self.course == "right":
                    return (0, 30), Mode.CARRY_BOTTLE2
                else:
                    return (30, 0), Mode.CARRY_BOTTLE2

            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_center={blue_center} | blue_pixel_count={blue_pixel_count}")
            blue_target_detected = (blue_center is not None and abs(blue_center[0] - 320) <= 100)

            if (not blue_target_detected) and (position_diff < 500):
                if self.course == "right":
                    return (0, 20), Mode.CARRY_BOTTLE2
                else:
                    return (20, 0), Mode.CARRY_BOTTLE2

            et.set_start_yaw()
            print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_target_detected={blue_target_detected} or position_diff={position_diff} >= 500 | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f} | blue_pixel_count={blue_pixel_count} | blue_center={blue_center}")
            phase.next_phase(current_pos)
            return (0, 0), Mode.CARRY_BOTTLE2

        # phase10: 青ターゲットを中心に合わせるだけ
        if phase.get_phase() == 10:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            
            if blue_center is not None and blue_center[1] > 10:
                # 距離計算して保存（フェーズ11で使用）
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | calc_blue_target_distance: X={blue_center[0]}, Y={blue_center[1]}, distance={calculated_distance}, pixels={blue_pixel_count}")
                
                # 中心に合わせる判定（±10ピクセル以内）
                if abs(blue_center[0] - self.center_x) <= 10:
                    # 中心に合った→次フェーズへ
                    et.set_start_yaw()
                    # フェーズ11開始位置を設定
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | centered | blue_center=({blue_center[0]}, {blue_center[1]}) | pixels={blue_pixel_count} | _calculated_distance={self._calculated_distance} | proceed to phase11")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE2
                else:
                    # 中心に向けて旋回
                    if blue_center[0] < self.center_x:
                        return (0, 5), Mode.CARRY_BOTTLE2  # 左旋回
                    else:
                        return (5, 0), Mode.CARRY_BOTTLE2  # 右旋回
            else:
                # 青ターゲットが見つからない→ヨー角調整
                in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
                if in_tolerance:
                    # ヨー角OK→フェーズ11へ
                    # フェーズ11開始位置を設定
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | no_blue_target | yaw_ok | proceed to phase11")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE2
                else:
                    # ヨー角調整
                    if yaw_error < 0:
                        return (5, 0), Mode.CARRY_BOTTLE2
                    else:
                        return (0, 5), Mode.CARRY_BOTTLE2

        # phase11: 青ターゲットのy座標が300になるまでゆっくり直進
        if phase.get_phase() == 11:
            # 標準的な距離計算を使用
            distance_from_start = phase.get_position_diff(current_pos)
            
            # 距離制限チェック - 計算距離に到達した場合は次フェーズへ
            if distance_from_start >= self._calculated_distance:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} >= _calculated_distance={self._calculated_distance} | proceed to tracking phase")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE2
            
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            
            # 青ターゲットが検出された場合の独立判定
            if blue_center is not None and blue_center[1] > 10:
                # 常に距離を計算して表示（統一フォーマット）
                calculated_distance = calc_blue_target_distance(blue_center)
                # print(f"[calc_blue_target_distance] X={blue_center[0]}, Y={blue_center[1]} → distance={calculated_distance} | pixels={blue_pixel_count}")
                
                # y座標が300以上になったら次フェーズへ（フェーズ11独立の距離計算）
                if blue_center[1] >= 300:
                    # _calculated_distanceは絶対にNoneにならない（デフォルト300保証済み）
                    if calculated_distance is not None:
                        self._calculated_distance = calculated_distance
                    # Noneの場合も既存の_calculated_distanceをそのまま使用（300または前回計算値）
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | blue_y={blue_center[1]} >= 300 | pixels={blue_pixel_count} | proceed to tracking phase | _calculated_distance={self._calculated_distance} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE2
                else:
                    # Y<300の場合は青ターゲットの中心に向けてcalc_motor_speedで進む（確立されたパターン）
                    target_x = blue_center[0]
                    accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                    left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=accelerated_speed)
                    return (left_speed, right_speed), Mode.CARRY_BOTTLE2
            else:
                # 青ターゲットが検出されない場合はyaw維持で直進
                accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
                # yaw維持直進コマンドを返す
                return (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # phase12: もう一度青ターゲットを中心に合わせる（EYE_BLUEのフェーズ2と同じ処理）
        if phase.get_phase() == 12:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            
            if blue_center is not None and blue_center[1] > 10:
                # 距離計算して保存（フェーズ13で使用）
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | calc_blue_target_distance: X={blue_center[0]}, Y={blue_center[1]}, distance={calculated_distance}, pixels={blue_pixel_count}")
                
                # 中心に合わせる判定（±5ピクセル以内）
                if abs(blue_center[0] - self.center_x) <= 5:
                    # 中心に合った→次フェーズへ
                    et.set_start_yaw()
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | centered | blue_center=({blue_center[0]}, {blue_center[1]}) | pixels={blue_pixel_count} | _calculated_distance={self._calculated_distance} | proceed to phase13")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE2
                else:
                    # 中心に向けて旋回
                    if blue_center[0] < self.center_x:
                        return (0, 5), Mode.CARRY_BOTTLE2  # 左旋回
                    else:
                        return (5, 0), Mode.CARRY_BOTTLE2  # 右旋回
            else:
                # 青ターゲットが見つからない→ヨー角調整
                in_tolerance, yaw_error = self.et.is_start_yaw_error_within(2.0)
                if in_tolerance:
                    # ヨー角OK→フェーズ13へ
                    print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | no_blue_target | yaw_ok | proceed to phase13")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.CARRY_BOTTLE2
                else:
                    # ヨー角調整
                    if yaw_error < 0:
                        return (5, 0), Mode.CARRY_BOTTLE2
                    else:
                        return (0, 5), Mode.CARRY_BOTTLE2

        # phase13: 青ターゲット追跡または計算距離まで直進。距離到達または青検出で次フェーズへ。
        if phase.get_phase() == 13:
            # 標準的な距離計算を使用
            distance_from_start = phase.get_position_diff(current_pos)
            # 計算距離到達で停止
            if distance_from_start >= self._calculated_distance:
                print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} >= _calculated_distance={self._calculated_distance} | STOP | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.CARRY_BOTTLE2
            # 青ターゲットの中心追従は行わず、常にyaw_straight_controlで直進
            accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
            left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
            start_yaw = et.get_start_yaw()
            current_yaw = et.get_yaw()
            yaw_error = current_yaw - start_yaw
            # print(f"[DEBUG] mode={Mode.CARRY_BOTTLE2.value} | phase={phase.get_phase()} | start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
            return (left_speed, right_speed), Mode.CARRY_BOTTLE2

        # 14. 状態リセットしBACK_AND_TURN2へ遷移
        if phase.get_phase() == 14:
            self.reset_action()
            return (0, 0), Mode.BACK_AND_TURN2

        print("[carry_bottle2_relative] Unexpected state reached.")
        return (0, 0), Mode.CARRY_BOTTLE2


    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.opposite_course)
            self._wait_start_time = None
        phase = self._phase
        current_pos = self.get_motor_position(self.opposite_course)

        # 0. 左モーターが抽象的な基準位置まで後退（両輪定速）。条件成立で次フェーズへ、モーター位置記録。
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 570:
                return (-30, -30), Mode.BACK_AND_TURN2
            print(f"[DEBUG] mode={Mode.BACK_AND_TURN2.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 570")
            phase.next_phase(current_pos)
            self._wait_start_time = time.time()
            return (0, 0), Mode.BACK_AND_TURN2

        # 1. 1秒待機フェーズ
        if phase.get_phase() == 1:
            if self._wait_start_time is None:
                self._wait_start_time = time.time()
            elapsed = time.time() - self._wait_start_time
            if elapsed >= 2.0:
                phase.next_phase(current_pos)
                self._wait_start_time = None
                return (0, 0), Mode.BACK_AND_TURN2
            # 待機中は停止
            return (0, 0), Mode.BACK_AND_TURN2

        # 2. 右旋回（抽象的な基準位置までは必ず旋回。条件成立後、ライン検出または基準位置到達まで所定速度で継続。条件成立で次フェーズへ）
        if phase.get_phase() == 2:
            position_diff = phase.get_position_diff(current_pos)
            horizontal_line_detected = is_upper_horizontal_line_detected(image)
            # 抽象的な基準位置までは必ず旋回
            if position_diff < 200:
                if self.course == "right":
                    return (30, 0), Mode.BACK_AND_TURN2
                else:
                    return (0, 30), Mode.BACK_AND_TURN2

            # ライン検出またはposition_diff>=400なら必ず即次フェーズへ
            if horizontal_line_detected or position_diff >= 400:
                print(f"[DEBUG] mode={Mode.BACK_AND_TURN2.value} | phase={phase.get_phase()} | horizontal_line_detected={horizontal_line_detected} | position_diff={position_diff} >= 400 | proceed to next phase")
                phase.next_phase(current_pos)
                return (0, 0), Mode.BACK_AND_TURN2

            # それ以外は継続
            if self.course == "right":
                return (20, 0), Mode.BACK_AND_TURN2
            else:
                return (0, 20), Mode.BACK_AND_TURN2

        # 3. 終了: 状態リセットし目標モードへ遷移（抽象化）
        if phase.get_phase() == 3:
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
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.HEAD_GOAL

        # 1. コース側モーターの移動距離が所定値未満なら中央追従、所定値以上で次フェーズへ遷移。到達でモーター位置記録。
        if phase.get_phase() == 1:
            threshold = 300
            position_diff = phase.get_position_diff(current_pos)
            if position_diff >= threshold:
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | position_diff={position_diff} >= {threshold}")
                phase.next_phase(current_pos)
            else:
                return (30, 30), Mode.HEAD_GOAL

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
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.HEAD_GOAL

        # 4. 青ライン検出後、コース側モーターの移動距離が所定値未満の間は左エッジトレース、到達でPAUSE（状態リセット）
        if phase.get_phase() == 4:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff >= 300:
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | position_diff={position_diff} | current_pos={current_pos} >= 300")
                phase.next_phase(current_pos)
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=30)
                return (left_speed, right_speed), Mode.HEAD_GOAL

        # 5. 青ライン検出後、コース側モーターの移動距離が所定値未満の間は直進、到達で次のフェーズへ
        if phase.get_phase() == 5:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff >= 200:
                print(f"[DEBUG] mode={Mode.HEAD_GOAL.value} | phase={phase.get_phase()} | position_diff={position_diff} | current_pos={current_pos} >= 200")
                phase.next_phase(current_pos)
            else:
                # go straight
                return (30, 30), Mode.HEAD_GOAL

        # 6. 所定距離到達で状態リセットしPAUSE
        if phase.get_phase() == 6:
            self.reset_action()
            return (0, 0), Mode.PAUSE

        print("[heading_goal_relative] Unexpected state reached.")
        return (0, 0), Mode.HEAD_GOAL

    def eye_blue(self, image: np.ndarray) -> Tuple[SpeedTuple, Mode]:

        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw()
            self._calculated_distance = 800 if self.course_type != "upper" else 300
            # phase0交互探索用の状態変数を初期化
            self._blue_search_cycle = 0
            self._blue_search_hold_count = 0
            self._blue_search_last_cmd = (0, 5)
        phase = self._phase
        current_pos = self.get_motor_position(self.course)
        et = self.et

        # phase0: 青ターゲット中心合わせ・積極探索
        if phase.get_phase() == 0:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            print(f"[DEBUG] find_blue_target_center: blue_center={blue_center}, blue_pixel_count={blue_pixel_count}")
            if blue_center is not None and blue_center[1] > 10:
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    print(f"[calc_blue_target_distance] X={blue_center[0]}, Y={blue_center[1]} → distance={calculated_distance} | pixels={blue_pixel_count}")
                diff = blue_center[0] - self.center_x
                print(f"[DEBUG] center_x={self.center_x}, blue_center_x={blue_center[0]}, diff={diff}")
                if abs(diff) <= 30:
                    et.set_start_yaw()
                    print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | centered | blue_center=({blue_center[0]}, {blue_center[1]}) | pixels={blue_pixel_count} | _calculated_distance={self._calculated_distance} | proceed to phase1")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.EYE_BLUE
                else:
                    if blue_center[0] < self.center_x:
                        print(f"[EYE_BLUE align] Blue left: turn left (0,5)")
                        return (0, 5), Mode.EYE_BLUE
                    else:
                        print(f"[EYE_BLUE align] Blue right: turn right (5,0)")
                        return (5, 0), Mode.EYE_BLUE
            else:
                # 青ターゲットが見つからない場合は左右交互に出力値5で10フレームずつ同じコマンドを維持
                N = 50  # コマンドを維持するフレーム数
                if self._blue_search_hold_count == 0:
                    self._blue_search_cycle += 1
                    if self._blue_search_cycle % 2 == 0:
                        self._blue_search_last_cmd = (0, 5)
                        print(f"[EYE_BLUE search] Blue not found: turn left (0,5) [hold {N} frames]")
                    else:
                        self._blue_search_last_cmd = (5, 0)
                        print(f"[EYE_BLUE search] Blue not found: turn right (5,0) [hold {N} frames]")
                    self._blue_search_hold_count = N
                self._blue_search_hold_count -= 1
                return self._blue_search_last_cmd, Mode.EYE_BLUE

        # phase1: 青ターゲットy>=300で次フェーズ。未満なら中心に向けて進む。
        if phase.get_phase() == 1:
            distance_from_start = phase.get_position_diff(current_pos)
            if distance_from_start >= self._calculated_distance:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} >= _calculated_distance={self._calculated_distance} | proceed to tracking phase")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_center is not None and blue_center[1] > 10:
                calculated_distance = calc_blue_target_distance(blue_center)
                print(f"[calc_blue_target_distance] X={blue_center[0]}, Y={blue_center[1]} → distance={calculated_distance} | pixels={blue_pixel_count}")
                if blue_center[1] >= 300:
                    if calculated_distance is not None:
                        self._calculated_distance = calculated_distance
                    print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | blue_y={blue_center[1]} >= 300 | pixels={blue_pixel_count} | proceed to tracking phase | _calculated_distance={self._calculated_distance} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.EYE_BLUE
                else:
                    target_x = blue_center[0]
                    accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                    left_speed, right_speed = self.calc_motor_speed(target_x, base_speed=accelerated_speed)
                    return (left_speed, right_speed), Mode.EYE_BLUE
            else:
                et.set_start_yaw()
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | no_blue_target | set_start_yaw | proceed to tracking phase | _calculated_distance={self._calculated_distance} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE

        # phase2: もう一度青ターゲット中心合わせ（carry_bottle2_relativeの最新ロジックに準拠）
        if phase.get_phase() == 2:
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            if blue_center is not None and blue_center[1] > 10:
                calculated_distance = calc_blue_target_distance(blue_center)
                if calculated_distance is not None:
                    self._calculated_distance = calculated_distance
                    print(f"[calc_blue_target_distance] X={blue_center[0]}, Y={blue_center[1]} → distance={calculated_distance} | pixels={blue_pixel_count}")
                if abs(blue_center[0] - self.center_x) <= 5:
                    et.set_start_yaw()
                    print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | centered | blue_center=({blue_center[0]}, {blue_center[1]}) | pixels={blue_pixel_count} | _calculated_distance={self._calculated_distance} | proceed to phase3")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.EYE_BLUE
                else:
                    if blue_center[0] < self.center_x:
                        return (0, 5), Mode.EYE_BLUE
                    else:
                        return (5, 0), Mode.EYE_BLUE
            else:
                in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
                if in_tolerance:
                    print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | no_blue_target | yaw_ok | proceed to phase3")
                    phase.next_phase(current_pos)
                    return (0, 0), Mode.EYE_BLUE
                else:
                    if yaw_error < 0:
                        return (5, 0), Mode.EYE_BLUE
                    else:
                        return (0, 5), Mode.EYE_BLUE

        # phase3: 青ターゲット追従または計算距離まで直進（carry_bottle2_relativeの最新ロジックに準拠）
        if phase.get_phase() == 3:
            distance_from_start = phase.get_position_diff(current_pos)
            if distance_from_start >= self._calculated_distance:
                print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} >= _calculated_distance={self._calculated_distance} | STOP | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.EYE_BLUE
            blue_center, _, blue_pixel_count = find_blue_target_center(image)
            print(f"[DEBUG] mode={Mode.EYE_BLUE.value} | phase={phase.get_phase()} | distance_from_start={distance_from_start} < {self._calculated_distance} | blue_center={blue_center} | pixels={blue_pixel_count}")
            if blue_pixel_count > 2000:
                if blue_center is not None:
                    et.set_start_yaw()
                    accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                    left_speed, right_speed = self.calc_motor_speed(blue_center[0], base_speed=accelerated_speed)
                else:
                    accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                    left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, adjust_speed=2, deadband=1)
                return (left_speed, right_speed), Mode.EYE_BLUE
            else:
                accelerated_speed = self.get_accelerated_base_speed(target_speed=10, acceleration_time=1.5)
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
                return (left_speed, right_speed), Mode.EYE_BLUE

        # phase4: 状態リセットしPAUSEへ遷移
        if phase.get_phase() == 4:
            self.reset_action()
            return (0, 0), Mode.PAUSE

        print("[eye_blue] Unexpected state reached.")
        return (0, 0), Mode.EYE_BLUE


# === テスト用メソッドはクラスの最後尾に追加 ===
    def test_mode_action(self, image=None) -> Tuple[Tuple[int, int], Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
            # 削除: self.et.reset_yaw()
            self.et.set_start_yaw()
            # あるべきヨー角（理想yaw）を管理
            self._ideal_yaw = self.et.get_start_yaw()
        phase = self._phase
        current_pos = self.get_motor_position(self.course)
        et = self.et

        # phase0: 直進（500進むまで）
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 500:
                accelerated_speed = self.get_accelerated_base_speed(target_speed=30, acceleration_time=1.5)
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
                return (left_speed, right_speed), Mode.TEST
            else:
                self._reset_acceleration_timer()
                print(f"[DEBUG] mode={Mode.TEST.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 500 | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.TEST

        # phase1: 90度旋回（左右切替、390進むまで）
        if phase.get_phase() == 1:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 390:
                if self.course == "right":
                    return (0, 20), Mode.TEST
                else:
                    return (20, 0), Mode.TEST
            else:
                # あるべきヨー角を基準に±90度
                if self.course == "right":
                    self._ideal_yaw -= 90.0
                else:
                    self._ideal_yaw += 90.0
                # -180～180でラップ
                self._ideal_yaw = et.wrap_angle(self._ideal_yaw)
                et.set_start_yaw(self._ideal_yaw)
                print(f"[DEBUG] mode={Mode.TEST.value} | phase={phase.get_phase()} | turn position_diff={position_diff} >= 390 | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.TEST

        # phase2: 旋回後のヨー角誤差調整（3秒間連続で誤差範囲内なら次フェーズ）
        if phase.get_phase() == 2:
            # 許容範囲に入ったら即フェーズ遷移
            in_tolerance, yaw_error = et.is_start_yaw_error_within(1.0)
            if in_tolerance:
                print(f"[DEBUG] mode={Mode.TEST.value} | phase={phase.get_phase()} | start_yaw_error={yaw_error:.2f} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.TEST
            else:
                if yaw_error < 0:
                    return (5, 0), Mode.TEST
                else:
                    return (0, 5), Mode.TEST

        # phase3: 直進（500進むまで）
        if phase.get_phase() == 3:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 500:
                accelerated_speed = self.get_accelerated_base_speed(target_speed=30, acceleration_time=1.5)
                left_speed, right_speed = et.yaw_straight_control(base_speed=accelerated_speed, deadband=1)
                return (left_speed, right_speed), Mode.TEST
            else:
                self._reset_acceleration_timer()
                print(f"[DEBUG] mode={Mode.TEST.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 500 | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.TEST

        # phase4: 90度旋回（左右切替、390進むまで）
        if phase.get_phase() == 4:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 390:
                if self.course == "right":
                    return (0, 20), Mode.TEST
                else:
                    return (20, 0), Mode.TEST
            else:
                if self.course == "right":
                    self._ideal_yaw -= 90.0
                else:
                    self._ideal_yaw += 90.0
                self._ideal_yaw = et.wrap_angle(self._ideal_yaw)
                et.set_start_yaw(self._ideal_yaw)
                print(f"[DEBUG] mode={Mode.TEST.value} | phase={phase.get_phase()} | turn position_diff={position_diff} >= 390 | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                phase.next_phase(current_pos)
                return (0, 0), Mode.TEST

        # phase5: 旋回後のヨー角誤差調整（3秒間連続で誤差範囲内なら次フェーズ）
        if phase.get_phase() == 5:
            # 許容範囲に入ったら即フェーズ遷移
            in_tolerance, yaw_error = et.is_start_yaw_error_within(1.0)
            if in_tolerance:
                phase_num = phase.get_phase()
                phase.next_phase(current_pos, skip=-5)  # 0に戻す
                # 0に戻すときは理想yawもリセット
                self._ideal_yaw = et.get_start_yaw()
                print(f"[DEBUG] mode={Mode.TEST.value} | phase={phase_num} | start_yaw_error={yaw_error:.2f} | start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f}")
                return (0, 0), Mode.TEST
            else:
                if yaw_error < 0:
                    return (5, 0), Mode.TEST
                else:
                    return (0, 5), Mode.TEST

        print("[test_mode] Unexpected state reached.")
        return (0, 0), Mode.TEST

