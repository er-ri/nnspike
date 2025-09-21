import time
from typing import Optional, Tuple
import numpy as np
from nnspike.unit.etrobot import ETRobot
from nnspike.constants import HIGH_SPEED_BASE
from nnspike.unit.action_chain import PhaseManager
from nnspike.constants import ROI_CNN
from nnspike.constants import Mode
from nnspike.constants import BASE_SPEED, ROI_COLOR2
from nnspike.utils import find_bottle_center

SpeedTuple = Tuple[int, int, int]
from nnspike.constants import Mode

class FastLapChain(object):

    """
    最速ラップタイムを記録するためのアクションチェーン管理クラス。
    ActionChainの設計・フェーズ管理を踏襲。
    各フェーズで直進・旋回・微調整・復帰などの動作を管理する。
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
        """
        アクション開始時の状態初期化処理。
        motor_side: "right"または"left"で初期位置記録対象を指定する。
        直進・旋回開始時に呼び出される。
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
        if not self._init:
            self.initialize_action(motor_side='right')
            self.et.set_start_yaw()
        phase = self._phase
        et = self.et

        # phase0: 左旋回中（yaw判定、90度到達で停止）
        if phase.get_phase() == 0:
            stop_turn = et.is_yaw_turn_finished(side="left", threshold_deg=90.0)
            if stop_turn:
                phase.next_phase()
                print(f"[TURN_LEFT] reached -90 deg and stopped | yaw={et.get_yaw():.2f}")
                return None, (0, 0, 0), Mode.TURN_LEFT_YAW
            else:
                print(f"[TURN_LEFT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                return None, (-30, 30, 0), Mode.TURN_LEFT_YAW

        # phase1: 左旋回後の微調整（±5度以内1秒静止でPAUSE）
        if phase.get_phase() == 1:
            if not hasattr(self, 'turn_left_reference_yaw') or self.turn_left_reference_yaw is None:
                self.turn_left_reference_yaw = et.get_yaw()
                self.turn_left_adjust_timer = time.time()
                self.turn_left_in_tolerance_time = None
            error = et.get_yaw() - self.turn_left_reference_yaw
            elapsed = time.time() - self.turn_left_adjust_timer if self.turn_left_adjust_timer is not None else 0
            if abs(error) <= 4.0:
                if self.turn_left_in_tolerance_time is None:
                    self.turn_left_in_tolerance_time = time.time()
                tolerance_elapsed = time.time() - self.turn_left_in_tolerance_time
                print(f"[TURN_LEFT][ADJUST][TOLERANCE] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, tolerance_elapsed={tolerance_elapsed:.2f}s")
                if tolerance_elapsed >= 2.0:
                    print(f"[TURN_LEFT][ADJUST][STOP] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, tolerance_elapsed={tolerance_elapsed:.2f}s")
                    self._phase = None
                    self.turn_left_reference_yaw = None
                    self.turn_left_adjust_timer = None
                    self.turn_left_in_tolerance_time = None
                    self.reset_action()
                    return None, None, Mode.PAUSE
            else:
                self.turn_left_in_tolerance_time = None
                if elapsed < 2.0:
                    if error < 0:
                        print(f"[TURN_LEFT][ADJUST] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                        return None, (20, -20, 0), Mode.TURN_LEFT_YAW
                    else:
                        print(f"[TURN_LEFT][ADJUST] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                        return None, (-20, 20, 0), Mode.TURN_LEFT_YAW
                else:
                    print(f"[TURN_LEFT][ADJUST][TIMEOUT] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_left_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                    self.turn_left_reference_yaw = None
                    self.turn_left_adjust_timer = None
                    self.turn_left_in_tolerance_time = None
                    self.reset_action()
                    return None, None, Mode.PAUSE

        # 速度返却（NoneでOK、et.set_motor_speedで直接制御）
        return None, None, Mode.TURN_LEFT_YAW

    def turn_right_yaw(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int, int]], 'Mode']:
        # ActionChain設計に厳密に合わせる: self._init判定→initialize_action→phase管理
        if not self._init:
            self.initialize_action(motor_side='left')
            self.et.set_start_yaw()
        phase = self._phase
        et = self.et

        # phase0: 右旋回中（yaw判定、90度到達で停止）
        if phase.get_phase() == 0:
            stop_turn = et.is_yaw_turn_finished(side="right", threshold_deg=90.0)
            if stop_turn:
                phase.next_phase()
                print(f"[TURN_RIGHT] reached +90 deg and stopped | yaw={et.get_yaw():.2f}")
                return None, (0, 0, 0), Mode.TURN_RIGHT_YAW
            else:
                print(f"[TURN_RIGHT] yaw={et.get_yaw():.2f}, yaw_start={et.get_start_yaw():.2f}, diff={et.get_yaw() - et.get_start_yaw():.2f}")
                return None, (30, -30, 0), Mode.TURN_RIGHT_YAW

        # phase1: 右旋回後の微調整（±5度以内1秒静止でPAUSE）
        if phase.get_phase() == 1:
            if not hasattr(self, 'turn_right_reference_yaw') or self.turn_right_reference_yaw is None:
                self.turn_right_reference_yaw = et.get_yaw()
                self.turn_right_adjust_timer = time.time()
                self.turn_right_in_tolerance_time = None
            error = et.get_yaw() - self.turn_right_reference_yaw
            elapsed = time.time() - self.turn_right_adjust_timer if self.turn_right_adjust_timer is not None else 0
            if abs(error) <= 4.0:
                if self.turn_right_in_tolerance_time is None:
                    self.turn_right_in_tolerance_time = time.time()
                tolerance_elapsed = time.time() - self.turn_right_in_tolerance_time
                print(f"[TURN_RIGHT][ADJUST][TOLERANCE] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_right_reference_yaw:.2f}, error={error:.2f}, tolerance_elapsed={tolerance_elapsed:.2f}s")
                if tolerance_elapsed >= 2.0:
                    print(f"[TURN_RIGHT][ADJUST][STOP] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_right_reference_yaw:.2f}, error={error:.2f}, tolerance_elapsed={tolerance_elapsed:.2f}s")
                    self._phase = None
                    self.turn_right_reference_yaw = None
                    self.turn_right_adjust_timer = None
                    self.turn_right_in_tolerance_time = None
                    self.reset_action()
                    return None, None, Mode.PAUSE
            else:
                self.turn_right_in_tolerance_time = None
                if elapsed < 2.0:
                    if error < 0:
                        print(f"[TURN_RIGHT][ADJUST] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_right_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                        return None, (20, -20, 0), Mode.TURN_RIGHT_YAW
                    else:
                        print(f"[TURN_RIGHT][ADJUST] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_right_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                        return None, (-20, 20, 0), Mode.TURN_RIGHT_YAW
                else:
                    print(f"[TURN_RIGHT][ADJUST][TIMEOUT] yaw={et.get_yaw():.2f}, ref_yaw={self.turn_right_reference_yaw:.2f}, error={error:.2f}, elapsed={elapsed:.2f}s")
                    self.turn_right_reference_yaw = None
                    self.turn_right_adjust_timer = None
                    self.turn_right_in_tolerance_time = None
                    self.reset_action()
                    return None, None, Mode.PAUSE

        # 速度返却（NoneでOK、et.set_motor_speedで直接制御）
        return None, None, Mode.TURN_RIGHT_YAW

    def fast_lap(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        if not self._init:
            # フェーズ0: course側モータ距離1000未満ならyaw_straight_controlで直進。1000以上で次フェーズ
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw()
            self.start_yaw = self.et.get_start_yaw()
        phase = self._phase
        et = self.et
        # start_yawはインスタンス変数として常に参照
        start_yaw = self.start_yaw

        # フェーズ0: 右モータ距離1000未満ならyaw_straight_controlで直進。1000以上で次フェーズ（直進区間）
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 1000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.FAST_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ1: 右旋回30度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yaw更新（右旋回区間）
        if phase.get_phase() == 1:
            stop_turn = et.is_yaw_turn_finished(side=self.course, threshold_deg=30.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw + 30.0)  # 右コースは+30度
                else:
                    self.et.set_start_yaw(start_yaw - 30.0)  # 左コースは-30度
            else:
                if self.course == "right":
                    return None, (100, 70, 0), Mode.FAST_LAP
                else:
                    return None, (70, 100, 0), Mode.FAST_LAP

        # フェーズ2: position_startとの差分500未満ならyaw_straight_control直進。500以上で次フェーズ、基準yaw更新（直進区間）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 500:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.FAST_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ3: 左旋回60度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yaw更新（左旋回区間）
        if phase.get_phase() == 3:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=60.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 30.0)  # 右コースは-30度
                else:
                    self.et.set_start_yaw(start_yaw + 30.0)  # 左コースは+30度
            else:
                if self.course == "right":
                    return None, (70, 100, 0), Mode.FAST_LAP
                else:
                    return None, (100, 70, 0), Mode.FAST_LAP

        # フェーズ4: position_startとの差分500未満ならyaw_straight_control直進。500以上で次フェーズ、基準yaw更新（直進区間）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 500:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.FAST_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ5: 右旋回30度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yaw更新（右旋回区間）
        if phase.get_phase() == 5:
            stop_turn = et.is_yaw_turn_finished(side=self.course, threshold_deg=30.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                self.et.set_start_yaw(start_yaw)
            else:
                if self.course == "right":
                    return None, (100, 70, 0), Mode.FAST_LAP
                else:
                    return None, (70, 100, 0), Mode.FAST_LAP

        # フェーズ6: position_startとの差分2000未満ならyaw_straight_control直進。2000以上で次フェーズ（直進区間）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 1000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.FAST_LAP
            else:
                phase.next_phase()
                return None, None, Mode.FAST_LAP

        # フェーズ7: 左旋回90度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yawをstart_yaw-90.0に更新
        if phase.get_phase() == 7:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=90.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 90.0)
                else:
                    self.et.set_start_yaw(start_yaw + 90.0)
            else:
                if self.course == "right":
                    return None, (70, 100, 0), Mode.FAST_LAP
                else:
                    return None, (100, 70, 0), Mode.FAST_LAP

        # フェーズ8: position_startとの差分1000未満ならyaw_straight_control直進。1000以上で次フェーズ
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 1000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.FAST_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ9: 左旋回90度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yawをstart_yaw-180.0に更新
        if phase.get_phase() == 9:
            stop_turn = et.is_yaw_turn_finished(side="left", threshold_deg=90.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 180.0)
                else:
                    self.et.set_start_yaw(start_yaw + 180.0)
            else:
                if self.course == "right":
                    return None, (70, 100, 0), Mode.FAST_LAP
                else:
                    return None, (100, 70, 0), Mode.FAST_LAP

        # フェーズ10: position_startとの差分2000未満ならyaw_straight_control直進。2000以上で次フェーズ
        if phase.get_phase() == 10:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 2000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.FAST_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ11: reset_action()してPAUSE復帰（ラップ終了）
        if phase.get_phase() == 11:
            self.reset_action()
            return None, None, Mode.PAUSE

        print("[FAST_LAP] Warning: Reached unexpected phase. Resetting action.")
        return None, None, Mode.FAST_LAP

    def shortcut_lap(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        if not self._init:
            # フェーズ0: course側モータ距離1000未満ならyaw_straight_controlで直進。1000以上で次フェーズ
            self.initialize_action(motor_side=self.course)
            self.et.set_start_yaw()
            self.start_yaw = self.et.get_start_yaw()
        phase = self._phase
        et = self.et
        # start_yawはインスタンス変数として常に参照
        start_yaw = self.start_yaw

        # フェーズ0: 右モータ距離1000未満ならyaw_straight_controlで直進。1000以上で次フェーズ（直進区間）
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 1000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.SHORTCUT_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ1: 右旋回30度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yaw更新（右旋回区間）
        if phase.get_phase() == 1:
            stop_turn = et.is_yaw_turn_finished(side="right", threshold_deg=30.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw + 30.0)  # 右コースは+30度
                else:
                    self.et.set_start_yaw(start_yaw - 30.0)  # 左コースは-30度
            else:
                if self.course == "right":
                    return None, (100, 70, 0), Mode.SHORTCUT_LAP
                else:
                    return None, (70, 100, 0), Mode.SHORTCUT_LAP

        # フェーズ2: position_startとの差分500未満ならyaw_straight_control直進。500以上で次フェーズ、基準yaw更新（直進区間）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 500:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.SHORTCUT_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ3: 左旋回60度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yaw更新（左旋回区間）
        if phase.get_phase() == 3:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=60.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 30.0)  # 右コースは-30度
                else:
                    self.et.set_start_yaw(start_yaw + 30.0)  # 左コースは+30度
            else:
                if self.course == "right":
                    return None, (70, 100, 0), Mode.FAST_LAP
                else:
                    return None, (100, 70, 0), Mode.FAST_LAP

        # フェーズ4: position_startとの差分2000未満ならyaw_straight_control直進。2000以上で次フェーズ、基準yaw更新（直進区間）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 2000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.SHORTCUT_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ5: 左旋回90度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yawをstart_yaw-90.0に更新
        if phase.get_phase() == 5:
            stop_turn = et.is_yaw_turn_finished(side="left", threshold_deg=60.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 90.0)
                else:
                    self.et.set_start_yaw(start_yaw + 90.0)
            else:
                if self.course == "right":
                    return None, (70, 100, 0), Mode.FAST_LAP
                else:
                    return None, (100, 70, 0), Mode.FAST_LAP

        # フェーズ6: position_startとの差分1000未満ならyaw_straight_control直進。1000以上で次フェーズ
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 1000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.SHORTCUT_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ7: 左旋回90度（is_yaw_turn_finished判定）。到達で次フェーズ、基準yawをstart_yaw-180.0に更新
        if phase.get_phase() == 7:
            stop_turn = et.is_yaw_turn_finished(side=self.opposite_course, threshold_deg=90.0)
            if stop_turn:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))
                if self.course == "right":
                    self.et.set_start_yaw(start_yaw - 180.0)
                else:
                    self.et.set_start_yaw(start_yaw + 180.0)
            else:
                if self.course == "right":
                    return None, (70, 100, 0), Mode.FAST_LAP
                else:
                    return None, (100, 70, 0), Mode.FAST_LAP

        # フェーズ8: position_startとの差分2000未満ならyaw_straight_control直進。2000以上で次フェーズ
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course)
            position_diff = abs(current_pos - position_start)
            if position_diff < 2000:
                left_speed, right_speed = et.yaw_straight_control(base_speed=HIGH_SPEED_BASE)
                return None, (left_speed, right_speed, 0), Mode.SHORTCUT_LAP
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course))

        # フェーズ9: reset_action()してPAUSE復帰（ラップ終了）
        if phase.get_phase() == 9:
            self.reset_action()
            return None, None, Mode.PAUSE
        
        print("[shortcut_lap] Unexpected state reached.")
        return None, None, Mode.SHORTCUT_LAP