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

    def initialize_action(self, motor_side: str = "right"):
        """
        アクション開始時の状態初期化処理。
        motor_side: "right"または"left"で初期位置記録対象を指定する。
        直進・旋回開始時に呼び出される。
        """
        self._phase = PhaseManager(motor_side)
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

    def get_accelerated_base_speed(self, elapsed_time: float, max_speed: Optional[int] = None) -> int:
        """
        改良された指数関数的滑らかな加速制御
        確実に最高速度100%に到達する設計
        
        Args:
            elapsed_time: 経過時間（秒）
            max_speed: 最高速度（デフォルト: HIGH_SPEED_BASE）
        
        Returns:
            適切なベース速度
        """
        if max_speed is None:
            max_speed = HIGH_SPEED_BASE
            
        # 0.8秒で確実に100%到達する設計
        if elapsed_time >= 0.8:
            return max_speed  # 確実に100%
            
        # 指数関数的加速（0-0.8秒）
        # k=4.0: より急峻な立ち上がり
        k = 4.0
        ratio = 1.0 - (2.71828 ** (-k * elapsed_time))
        
        # 最低速度15%、最大95%の範囲で制御
        min_ratio = 0.15
        max_ratio = 0.95  # 0.8秒で95%到達
        final_ratio = min_ratio + (max_ratio - min_ratio) * ratio
        
        return int(max_speed * final_ratio)

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
            in_tolerance, yaw_error = et.is_start_yaw_error_within(1.0)
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
            in_tolerance, yaw_error = et.is_start_yaw_error_within(1.0)
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
        phase = self._phase
        et = self.et
        current_pos = self.get_motor_position(self.course)
        start_yaw = self.start_yaw

        # フェーズ0: course側モータ距離3000未満ならyaw_straight_controlで直進。3000以上で次フェーズ
        if phase.get_phase() == 0:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 3000:
                # 汎用的な段階的加速制御メソッドを使用
                elapsed_time = time.time() - self.lap_start_time
                base_speed = self.get_accelerated_base_speed(elapsed_time, HIGH_SPEED_BASE)
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

            color_info = self.get_color_sensor_values()
            if color_info["color_type"] == "white":
                self._phase2_color_count += 1
            else:
                self._phase2_color_count = 0
            # 連続色2回以上で次フェーズ
            threshold = 2600
            print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | position_diff={position_diff} | color={color_info['color']} | color_type={color_info['color_type']} | color_count={self._phase2_color_count} | current_pos={current_pos}")
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

        # フェーズ6: position_startとの差分2000未満ならyaw_straight_control直進。2000以上で次フェーズ
        if phase.get_phase() == 6:
            position_diff = phase.get_position_diff(current_pos)
            if position_diff < 300:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return (left_speed, right_speed), Mode.FAST_LAP
            else:
                # ラップ終了タイム記録
                self.lap_end_time = time.time()
                lap_elapsed = self.lap_end_time - self.lap_start_time
                print(f"[DEBUG] mode={Mode.FAST_LAP.value} | phase={phase.get_phase()} | position_diff={position_diff} >= 300 | current_pos={current_pos} | time: {lap_elapsed:.3f}秒")
                phase.next_phase(current_pos)
                return (0, 0), Mode.FAST_LAP

        # フェーズ7: reset_action()してPAUSE復帰（ラップ終了）
        if phase.get_phase() == 7:
            self.reset_action()
            # return (0, 0), Mode.PAUSE
            return (0, 0), Mode.DOUBLE_LOOP

        print("[FAST_LAP] Unexpected state reached.")
        return (0, 0), Mode.FAST_LAP
