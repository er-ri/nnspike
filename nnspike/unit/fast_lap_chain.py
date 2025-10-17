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
        # Simplified 3-phase fast_lap:
        # phase0: straight (existing start-dash behavior) until threshold
        # phase1: small right turn (~12deg)
        # phase2: back same distance as phase0
        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw()
            self.start_yaw = self.et.get_start_yaw()
            self.lap_start_time = time.time()
            self._prev_color = None
            self._phase0_distance = 0
        phase = self._phase
        et = self.et
        current_pos = self.get_motor_position(self.course)

        # PHASE 0: same as before - straight until position_diff >= 3000
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 3000:
                base_speed = self.get_fast_lap_start_dash_speed(HIGH_SPEED_BASE, 0.5)
                left_speed, right_speed = et.yaw_straight_control(base_speed=base_speed)
                return (left_speed, right_speed), Mode.FAST_LAP
            else:
                # record distance covered in phase0 for later backing
                self._phase0_distance = position_diff
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase=0->1 | position_diff={position_diff} | time: {lap_elapsed:.3f}s")
                phase.next_phase(current_pos)

        # PHASE 1: small right turn (~12deg). drive until yaw turn finished
        if phase.get_phase() == 1:
            stop_turn = et.is_yaw_turn_finished(side="right", threshold_deg=12.0)
            if stop_turn:
                # update reference yaw
                et.set_start_yaw(et.get_start_yaw() + 12.0)
                lap_elapsed = time.time() - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase=1->2 | set_start_yaw={et.get_start_yaw():.2f} | time: {lap_elapsed:.3f}s")
                phase.next_phase(current_pos)
            else:
                # gentle right turn: left slightly faster than right
                return (100, 80), Mode.FAST_LAP

        # PHASE 2: back the same distance as phase0
        if phase.get_phase() == 2:
            # when entering phase2, position_start is already set to current_pos by next_phase
            back_diff = phase.get_position_diff(current_pos)
            target = int(self._phase0_distance)
            print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase=2 | back_diff={back_diff} | target_back={target}")
            if back_diff < target:
                # drive backwards with moderate speed
                return (-HIGH_SPEED_BASE, -HIGH_SPEED_BASE), Mode.FAST_LAP
            else:
                # finished backing
                self.reset_action()
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase=2 finished | backed {back_diff} >= {target}")
                return (0, 0), Mode.PAUSE

        # fallback
        print("[FAST_LAP] Unexpected state reached.")
        return (0, 0), Mode.FAST_LAP
