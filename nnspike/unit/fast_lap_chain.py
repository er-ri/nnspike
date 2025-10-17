from multiprocessing import current_process
import time
from typing import Optional, Tuple
import numpy as np
from nnspike.unit.etrobot import ETRobot
from nnspike.constants import HIGH_SPEED_BASE
from nnspike.unit.action_chain import PhaseManager
from nnspike.constants import ROI_CNN, ROI_LINE_CORNER
from nnspike.constants import Mode

from nnspike.utils.control import is_fast_corner_detected

SpeedTuple = Tuple[int, int, int]
from nnspike.constants import Mode

class FastLapChain(object):

    """
    最速ラップタイムを記録するためのアクションチェーン管理クラス。
    ActionChainの設計・フェーズ管理を踏襲。
    各フェーズで直進・旋回・微調整・復帰などの動作を管理する。
    """
    def __init__(self, et: ETRobot, course: str) -> None:
        """ActionChainの初期化処理."""
        self.et = et  # ロボット本体
        self.course = course  # コース種別
        # コース種別の逆コースを定義
        if course == "right":
            self.opposite_course = "left"
        else:
            self.opposite_course = "right"
        self._init = False
        # 加速制御用プライベート変数
        self._acceleration_start_time = None
        # パワー監視・加速制御用プライベート変数
        self._prev_motor_a_power = 0
        self._prev_motor_b_power = 0
        self._freeze_speed = False
        self._last_speed = 40

    def initialize_action(self, motor_side: str = "right") -> None:
        """
        アクション開始時の状態初期化処理。
        motor_side: "right"または"left"で初期位置記録対象を指定する。
        直進・旋回開始時に呼び出される。
        """
        self._phase = PhaseManager(motor_side)
        self._phase.set_position_start("position_start", self.get_motor_position(motor_side))
        self._init = True
        # アクション開始時に加速タイマーをリセット
        self._reset_acceleration_timer()

    def reset_action(self) -> None:
        """アクション終了時の状態リセット処理."""
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

    def get_fast_lap_start_dash_speed(self, target_speed: int = HIGH_SPEED_BASE, acceleration_time: float = 0.5) -> int:
        """
        ファストラップ専用スタートダッシュ加速制御。
        action_chainの加速とは完全に独立した、より積極的な加速を提供する。
        motor_a_power/motor_b_powerの落ち込みを監視し、落ち込んだらスピード上昇を一時停止、回復したら再加速。
        最低パワーは40から開始。
        """
        # タイマーが未初期化の場合は自動開始
        if self._acceleration_start_time is None:
            self._start_acceleration_timer()

        # 経過時間を計算
        elapsed_time = self._get_acceleration_elapsed_time()

        # acceleration_time秒以降は絶対にtarget_speedを返す（余計な計算・判定なし）
        if elapsed_time >= acceleration_time:
            return target_speed

        # --- acceleration_time未満のみ従来の加速・freeze判定を行う ---
        left_power, right_power = self.et.get_motor_power()
        motor_a_power = right_power  # A=right
        motor_b_power = left_power   # B=left

        # パワー落ち込み判定（絶対値で前回より下がったらfreeze、上がったら解除）
        if (abs(motor_a_power) < abs(self._prev_motor_a_power) or
            abs(motor_b_power) < abs(self._prev_motor_b_power)):
            self._freeze_speed = True
        elif (abs(motor_a_power) >= abs(self._prev_motor_a_power) or
              abs(motor_b_power) >= abs(self._prev_motor_b_power)):
            self._freeze_speed = False

        ratio = elapsed_time / acceleration_time
        aggressive_ratio = ratio ** 1.5
        min_speed = 40
        calculated_speed = int(min_speed + (target_speed - min_speed) * aggressive_ratio)
        speed = max(min_speed, min(calculated_speed, target_speed))


        if self._freeze_speed:
            speed = self._last_speed

        # 状態更新
        self._last_speed = speed
        self._prev_motor_a_power = motor_a_power
        self._prev_motor_b_power = motor_b_power

        return speed

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

    def turn_left_yaw(self, image: np.ndarray) -> Tuple[Tuple[int, int], Mode]:
        # ActionChain設計に厳密に合わせる: self._init判定→initialize_action→phase管理
        if not self._init:
            self.initialize_action(motor_side='right')
            self.et.set_start_yaw()
        phase = self._phase
        et = self.et
        current_pos = self.get_motor_position(self.course)
        # デバッグ: 呼び出し時の yaw と start_yaw を常に出力
        try:
            _yaw = et.get_yaw()
            _yaw_s = f"{_yaw:.2f}"
        except Exception:
            _yaw_s = "None"
        try:
            _start_yaw = et.get_start_yaw()
            _start_s = f"{_start_yaw:.2f}"
        except Exception:
            _start_s = "None"
        print(f"[DEBUG_YAW][turn_left_yaw] phase={phase.get_phase()} yaw={_yaw_s} start_yaw={_start_s} current_pos={current_pos}")
        # 表示後に改行
        print()

        # phase0: 左旋回中（yaw判定、90度到達で停止）
        if phase.get_phase() == 0:
            stop_turn = et.is_yaw_turn_finished(side="left", threshold_deg=90.0)
            if stop_turn:
                phase.next_phase(current_pos)
                print(f"[TURN_LEFT] reached -90 deg and stopped | yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                et.set_start_yaw(et.get_start_yaw() - 90.0)
                # et.set_start_yaw_nearest_horizontal_pole()  
                return (0, 0), Mode.TURN_LEFT_YAW
            else:
                print(f"[TURN_LEFT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                return (0, 30), Mode.TURN_LEFT_YAW

        # phase1: 左旋回後の微調整（±4度以内2秒静止でPAUSE）
        if phase.get_phase() == 1:
            # carry_bottle1_relativeのphase2と完全同一ロジック
            in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
            start_yaw = et.get_start_yaw()
            current_yaw = et.get_yaw()
            if in_tolerance:
                print(f"[TURN_LEFT][ADJUST][TOLERANCE] in_tolerance={in_tolerance} | start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
                phase.next_phase(current_pos)
            else:
                print(f"[TURN_LEFT][ADJUST][CORRECT] start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
                if yaw_error < 0:
                    return (5, 0), Mode.TURN_LEFT_YAW
                else:
                    return (0, 5), Mode.TURN_LEFT_YAW

        # phase2のみポーズ復帰＋リセット
        if phase.get_phase() == 2:
            self.reset_action()
            return (0, 0), Mode.PAUSE

        # 速度返却（NoneでOK、et.set_motor_speedで直接制御）
        return (0, 0), Mode.TURN_LEFT_YAW

    def turn_right_yaw(self, image: np.ndarray) -> Tuple[Tuple[int, int], Mode]:
        # ActionChain設計に厳密に合わせる: self._init判定→initialize_action→phase管理
        if not self._init:
            self.initialize_action(motor_side='left')
            self.et.set_start_yaw()
        phase = self._phase
        et = self.et
        current_pos = self.get_motor_position(self.course)
        # デバッグ: 呼び出し時の yaw と start_yaw を常に出力
        try:
            _yaw = et.get_yaw()
            _yaw_s = f"{_yaw:.2f}"
        except Exception:
            _yaw_s = "None"
        try:
            _start_yaw = et.get_start_yaw()
            _start_s = f"{_start_yaw:.2f}"
        except Exception:
            _start_s = "None"
        print(f"[DEBUG_YAW][turn_right_yaw] phase={phase.get_phase()} yaw={_yaw_s} start_yaw={_start_s} current_pos={current_pos}")
        # 表示後に改行
        print()

        # phase0: 右旋回中（yaw判定、90度到達で停止）
        if phase.get_phase() == 0:
            stop_turn = et.is_yaw_turn_finished(side="right", threshold_deg=90.0)
            if stop_turn:
                phase.next_phase(current_pos)
                print(f"[TURN_RIGHT] reached +90 deg and stopped | yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                et.set_start_yaw(et.get_start_yaw() + 90.0)
                # et.set_start_yaw_nearest_horizontal_pole()  
                return (0, 0), Mode.TURN_RIGHT_YAW
            else:
                print(f"[TURN_RIGHT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                return (30, 0), Mode.TURN_RIGHT_YAW

        # phase1: 右旋回後の微調整（±4度以内2秒静止でPAUSE）
        if phase.get_phase() == 1:
            # carry_bottle1_relativeのphase2と完全同一ロジック
            in_tolerance, yaw_error = et.is_start_yaw_error_within(2.0)
            start_yaw = et.get_start_yaw()
            current_yaw = et.get_yaw()
            if in_tolerance:
                print(f"[TURN_RIGHT][ADJUST][TOLERANCE] in_tolerance={in_tolerance} | start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
                phase.next_phase(current_pos)
            else:
                print(f"[TURN_RIGHT][ADJUST][CORRECT] start_yaw={start_yaw:.2f} | current_yaw={current_yaw:.2f} | yaw_error={yaw_error:.2f}")
                if yaw_error < 0:
                    return (5, 0), Mode.TURN_RIGHT_YAW
                else:
                    return (0, 5), Mode.TURN_RIGHT_YAW

        # phase2のみポーズ復帰＋リセット
        if phase.get_phase() == 2:
            self.reset_action()
            return (0, 0), Mode.PAUSE

        # 速度返却（NoneでOK、et.set_motor_speedで直接制御）
        return (0, 0), Mode.TURN_RIGHT_YAW

    def fast_lap(self, image: np.ndarray) -> Tuple[Tuple[int, int], Mode]:
        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw()
            self.start_yaw = self.et.get_start_yaw()
            self.lap_start_time = time.time()
            self._phase2_color_count = 0  # フェーズ2色検出カウンタ
            self._prev_color = None  # 前回の色値を初期化
        phase = self._phase
        et = self.et
        current_pos = self.get_motor_position(self.course)
        start_yaw = self.start_yaw
        # デバッグ: 呼び出し時の yaw と start_yaw を常に出力
        try:
            _yaw = et.get_yaw()
            _yaw_s = f"{_yaw:.2f}"
        except Exception:
            _yaw_s = "None"
        try:
            _start_yaw = et.get_start_yaw()
            _start_s = f"{_start_yaw:.2f}"
        except Exception:
            _start_s = "None"
        print(f"[DEBUG_YAW][fast_lap] phase={phase.get_phase()} yaw={_yaw_s} start_yaw_obj={_start_s} start_yaw_local={start_yaw} current_pos={current_pos}")
        # 表示後に改行
        print()

        # フェーズ0: course側モータ距離3000未満ならyaw_straight_controlで直進。3000以上で次フェーズ
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 3000:
                # ファストラップ専用スタートダッシュ加速制御メソッドを使用
                base_speed = self.get_fast_lap_start_dash_speed(HIGH_SPEED_BASE, 0.5)
                left_speed, right_speed = et.yaw_straight_control(base_speed=base_speed)
                return (left_speed, right_speed), Mode.FAST_LAP
            else:
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 3000 | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)

        # フェーズ1: 左旋回20度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yaw更新
        if phase.get_phase() == 1:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=12.0)
            if stop_turn:
                if self.course == "right":
                    et.set_start_yaw(start_yaw - 12.0)
                else:
                    et.set_start_yaw(start_yaw + 12.0)
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f} | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)
            else:
                if self.course == "right":
                    return (80, 100), Mode.FAST_LAP
                else:
                    return (100, 80), Mode.FAST_LAP

        # フェーズ2: position_diffが1500未満なら直進、2000以上なら強制で次フェーズ、それ以外は従来通り
        if phase.get_phase() == 2:
            position_diff = phase.get_position_diff(current_pos)
            # 1500未満は無条件で直進
            if position_diff < 2400:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return (left_speed, right_speed), Mode.FAST_LAP

            # 2400を超えたら色差計測開始
            color_info = self.get_color_sensor_values()
            current_color = color_info["color"]
            
            # 前回の色値が記録されていない場合は初期化
            if self._prev_color is None:
                self._prev_color = current_color
                self._phase2_color_count = 0
                color_diff = 0
            else:
                # 色差を計算
                color_diff = abs(current_color - self._prev_color)
                
                # 色差が150を超える場合にカウント
                if color_diff > 150:
                    self._phase2_color_count += 1
                    self._prev_color = current_color  # 色値更新
                
            # 連続色差2回以上で次フェーズ
            threshold = 2600
            print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | position_diff={position_diff} | color={current_color} | color_diff={color_diff} | color_count={self._phase2_color_count} | current_pos={current_pos}")
            if self._phase2_color_count >= 2 or position_diff >= threshold:
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | color_count={self._phase2_color_count} >= 2 or position_diff={position_diff} >= {threshold} | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)
                self._phase2_color_count = 0
            else:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return (left_speed, right_speed), Mode.FAST_LAP

        # フェーズ3: 左旋回90度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yawをstart_yaw-90.0に更新
        if phase.get_phase() == 3:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=78.0)
            if stop_turn:
                if self.course == "right":
                    et.set_start_yaw(start_yaw - 90.0)
                else:
                    et.set_start_yaw(start_yaw + 90.0)
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | set_start_yaw={et.get_start_yaw():.2f} | current_yaw={et.get_yaw():.2f} | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)
            else:
                if self.course == "right":
                    return (70, 100), Mode.FAST_LAP
                else:
                    return (100, 70), Mode.FAST_LAP

        # フェーズ4: 最小距離未満は何も判定せず直進。最小距離以上でcorner判定・閾値判定。
        if phase.get_phase() == 4:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 500:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return (left_speed, right_speed), Mode.FAST_LAP

            fast_corner = is_fast_corner_detected(image, roi=ROI_LINE_CORNER, course=self.course)
            if fast_corner or position_diff >= 1800:
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | fast_corner_detected={fast_corner} | position_diff={position_diff} | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)
            else:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return (left_speed, right_speed), Mode.FAST_LAP

        # フェーズ5: 左旋回90度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yawをstart_yaw-180.0に更新
        if phase.get_phase() == 5:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=90.0)
            if stop_turn:
                phase.next_phase(current_pos)
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | yaw={self.et.get_yaw():.2f} | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 180.0)
                else:
                    self.et.set_start_yaw(start_yaw + 180.0)
            else:
                if self.course == "right":
                    return (75, 100), Mode.FAST_LAP
                else:
                    return (100, 75), Mode.FAST_LAP

        # フェーズ6: position_startとの差分200未満ならyaw_straight_control直進。200以上で次フェーズ
        if phase.get_phase() == 6:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 200:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return (left_speed, right_speed), Mode.FAST_LAP
            else:
                # ラップ終了タイム記録
                self.lap_end_time = time.time()
                lap_elapsed = self.lap_end_time - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 200 | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)
                return (0, 0), Mode.FAST_LAP

        # フェーズ7: reset_action()してPAUSE復帰（ラップ終了）
        if phase.get_phase() == 7:
            self.reset_action()
            # return (0, 0), Mode.PAUSE
            return (0, 0), Mode.DOUBLE_LOOP

        print("[FAST_LAP] Unexpected state reached.")
        return (0, 0), Mode.FAST_LAP
