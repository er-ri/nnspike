import time  # 時間計測用
from typing import Optional, Tuple  # 型ヒント用

import numpy as np  # 画像処理用

# 定数・モード・ROI設定
from nnspike.constants import OFFSET_Y, ROI_CNN, Mode, BASE_SPEED
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
    is_horizontal_black_line_detected,  # 水平黒ライン検出
    is_vertical_black_line_detected,  # 垂直黒ライン検出
    is_general_horizontal_line_detected,  # 一般的な水平黒ライン検出
)

class PhaseManager:

    def __init__(self):
        self._state = {}
        self._state["phase"] = 0
        self._state["position_start"] = None

    def get_phase(self) -> int:
        return self._state.get("phase", 0)

    def next_phase(self) -> None:
        self._state["phase"] = self._state.get("phase", 0) + 1

    def set_position_start(self, key: str, value) -> None:
        """
        指定したkey（例: 'right_position_start'）にvalue（例: モーター位置）をセット。
        valueがint型以外の場合は0に変換してセット。
        """
        if not isinstance(value, int):
            value = 0
        self._state[key] = value

    def get_position_start(self, key: str) -> int:
        """
        指定したkeyのposition_start値を取得。未設定やint型以外なら0を返す。
        """
        value = self._state.get(key, None)
        if not isinstance(value, int):
            return 0
        return value

class ActionChain(object):
    """
    ETRobotのためのアクションシーケンス管理クラス。

    各アクション（左旋回・右旋回・短時間旋回・青ボトルキャッチ等）を、
    指定時間またはモーター相対位置・画像認識条件で状態遷移しながら実行する。
    状態管理はself._stateのdictで行い、各アクションはフェーズごとに分岐。
    コメント・docstringは必ず実装内容と一致させること。
    """

    def __init__(self, et: ETRobot, course: str) -> None:
        self.et = et  # ロボット本体
        self.course = course  # コース種別
        self.start_time = 0.0  # アクション開始時刻
        self.current_time = 0.0  # 現在時刻
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN  # ROI座標
        self._init = False  
        
    def get_motor_position(self, side: str = "right", mode: str = "position", status=None) -> int:
        """
        side='right'で右モータ(B)、'left'で左モータ(A)のrelative_positionを返す。
        mode='position'なら該当モータのrelative_position（絶対値, Noneなら0）、'status'ならstatusオブジェクト。
        status引数を指定すればそれを使い、未指定時のみ内部で取得する。
        負荷軽減のため、複数回呼び出し時はstatusを外部で取得・使い回すこと。
        ただしmode='status'時は必ず最新statusを再取得する。
        """
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
            motor_key = "B" if side == "right" else "A"
            if status is not None and status.motors.get(motor_key) is not None:
                pos = status.motors[motor_key].relative_position
                if isinstance(pos, int):
                    return abs(pos)
                else:
                    print(f"[get_motor_position] {motor_key} position invalid: {pos} (return 0)")
                    return 0
            print(f"[get_motor_position] status or motor_key invalid (return 0)")
            return 0

    def turn_left(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        左旋回アクション。
        1.5秒間、左モータ:0・右モータ:30で旋回し、1.5秒経過後にPAUSEへ遷移。
        戻り値: (None, (左速度, 右速度), モード)
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 1.5:
            return None, (0, 30), Mode.TURN_LEFT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def trun_right(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        右旋回アクション。
        1.5秒間、左モータ:30・右モータ:0で旋回し、1.5秒経過後にPAUSEへ遷移。
        戻り値: (None, (左速度, 右速度), モード)
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 1.5:
            return None, (30, 0), Mode.TURN_RIGHT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def small_turn_left(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        短時間（0.3秒）左旋回アクション。
        0.3秒間、左モータ:0・右モータ:50で旋回し、0.3秒経過後にPAUSEへ遷移。
        戻り値: (None, (左速度, 右速度), モード)
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.3:
            return None, (0, 50), Mode.SMALL_TURN_LEFT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def small_turn_right(self) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        短時間（0.3秒）右旋回アクション。
        0.3秒間、左モータ:50・右モータ:0で旋回し、0.3秒経過後にPAUSEへ遷移。
        戻り値: (None, (左速度, 右速度), モード)
        """
        self.start_time = time.time() if self.start_time == 0.0 else self.start_time
        self.current_time = time.time()

        elapsed_time = self.current_time - self.start_time
        if elapsed_time < 0.3:
            return None, (50, 0), Mode.SMALL_TURN_RIGHT
        self.start_time = 0.0
        return None, None, Mode.PAUSE

    def blue_bottle_catch(self, image: np.ndarray) -> tuple:
        """
        青ボトルキャッチモード。
        ・phase0: 青ターゲット中心x座標へ追従（青ピクセル数1000超えたらphase1へ）
        ・phase1: 青ピクセル数1000以上の間は中心x座標へ追従、500以下でphase2へ（右モーター位置記録）
        ・phase2: 青ピクセル数500以下になってから右モーター300ユニット移動まで中心x座標へ追従、300到達でphase3へ
        ・phase3: 状態リセットしPAUSEへ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        # 初回呼び出し時のみ初期化
        # phase0: 初期化（PhaseManager生成、右モーター初期位置記録）
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            self._phase.set_position_start("position_start", self.get_motor_position('right', status=self._status))
            self._init = True
        phase = self._phase
        status = self._status

        # phase0: 青ターゲット中心x座標へ追従（青ピクセル数1000超えたらphase1へ）
        if phase.get_phase() == 0:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                phase.next_phase()
            return target_x, None, Mode.BLUE_BOTTLE_CATCH

        # phase1: 青ピクセル数1000以上の間は中心x座標へ追従、500以下でphase2へ（右モーター位置記録）
        if phase.get_phase() == 1:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                phase.next_phase()
                # phase2用 右モーター相対位置記録
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            return target_x, None, Mode.BLUE_BOTTLE_CATCH

        # phase2: 青ピクセル数500以下になってから右モーター300ユニット移動まで中心x座標へ追従、300到達でphase3へ
        if phase.get_phase() == 2:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 300:
                return target_x, None, Mode.BLUE_BOTTLE_CATCH
            else:
                phase.next_phase()
        # phase3: 状態リセットしPAUSEへ遷移
        if phase.get_phase() == 3:
            self._init = False
            return None, None, Mode.PAUSE

    def turn_left_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        左旋回（右モーターBの相対位置差分で判定）。430未満の間は左:0,右:30で継続。430超えたらPAUSE。
        """
        status = self.get_motor_position(mode="status")
        # 右モーターの初期位置を記録
        if not hasattr(self, '_right_position_start') or self._right_position_start is None:
            self._right_position_start = self.get_motor_position('right', status=status)

        right_position = self.get_motor_position('right', status=status)
        # 右(B)の開始～現在の差分が430を超えたら停止
        if abs(right_position - self._right_position_start) > 430:
            self._right_position_start = 0
            return None, None, Mode.PAUSE
        return None, (0, 30), Mode.TURN_LEFT_RELATIVE

    def turn_right_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        右旋回（左モーターAの相対位置差分で判定）。430未満の間は左:30,右:0で継続。430超えたらPAUSE。
        """
        status = self.get_motor_position(mode="status")
        # 左モーターの初期位置を記録
        if not hasattr(self, '_left_position_start') or self._left_position_start is None:
            self._left_position_start = self.get_motor_position('left', status=status)

        left_position = self.get_motor_position('left', status=status)
        # 左(A)の開始～現在の差分が430を超えたら停止
        if abs(left_position - self._left_position_start) > 430:
            self._left_position_start = 0
            return None, None, Mode.PAUSE
        return None, (30, 0), Mode.TURN_RIGHT_RELATIVE

    def avoid_obstacle_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        avoid_obstacleの位置判定バージョン。
        以下の順で動作する:
        0. 左旋回（右モーター500ユニット移動まで, 左:40, 右:70）
        1. 右旋回（右モーター650ユニット移動まで, 左:80, 右:50）
        2. 左旋回（右モーター350ユニット移動まで, 左:40, 右:70）
        3. チェーン終了で右端追従モードへ復帰
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            # 初期位置記録を初回初期化時に実施
            self._phase.set_position_start("position_start", self.get_motor_position('right', status=self._status))
            self._init = True
        phase = self._phase
        status = self._status

        # 0. 左旋回（右モーター500ユニット移動まで, 左:40, 右:70）
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 500:
                return None, (40, 70), Mode.AVOID_OBSTACLE
            else:
                # 500ユニット到達したので次のフェーズへ
                phase.next_phase()
                # phase1用も右モーター相対位置記録（絶対値）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            return None, (40, 70), Mode.AVOID_OBSTACLE

        # 1. 右旋回（右モーター700ユニット移動まで, 左:80, 右:50）
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 550:
                return None, (70, 40), Mode.AVOID_OBSTACLE
            else:
                # 650ユニット到達したので次のフェーズへ
                phase.next_phase()
                # phase2用も右モーター相対位置記録（絶対値）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 2. 左旋回（右モーター350ユニット移動まで, 左:40, 右:70）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            distance = abs(current_pos - position_start)
            # 走行距離が200未満なら常に(40,70)で走行
            if distance < 250:
                return None, (40, 70), Mode.AVOID_OBSTACLE
            # 250以上350未満の間はis_vertical_black_line_detected(image)がTrueなら即フェーズ3へ
            elif distance < 350:
                if is_vertical_black_line_detected(image):
                    phase.next_phase()
                    # すぐ次の処理でphase3に入る
                else:
                    return None, (40, 70), Mode.AVOID_OBSTACLE
            # 450以上なら強制的にフェーズ3へ
            else:
                phase.next_phase()

        # 3. チェーン終了
        if phase.get_phase() == 3:
            # 状態リセット
            self._init = False
            return None, None, Mode.FOLLOW_RIGHT_EDGE

# --- 以下、*_relativeメソッド（元メソッド完全コピー） ---

    def carry_bottle1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        carry_bottle1の位置判定バージョン。
        以下の順で動作する:
        0. 右エッジトレース（赤ピクセル数が3000を超えたらphase1へ）
        1. 赤ボトル中心追従（右モーター1000ユニット移動まで、赤ピクセル500未満なら中央）
        2. 左エッジトレース（右モーター300ユニット移動まで）
        3. 左旋回（右モーター400ユニット移動まで, 左:0, 右:30）
        4. 直進（右モーター300ユニット移動まで）
        5. 仮想ライン直進（右モーター800ユニット移動まで, get_virtual_line_target_x, previous_center_x=pre_target_x）
        6. 直進（右モーター1900ユニット移動まで）
        7. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最低300・最大右モーター500ユニット）
        8. 青検出（1000超えたらphase9へ）
        9. 青1000以上の間center追従、500以下でphase10へ
        10. 青500以下になってから右モーター300ユニット移動まで center追従、その後BACK_AND_TURN1
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            # 初期位置記録を初回初期化時に実施
            self._phase.set_position_start("position_start", self.get_motor_position('right', status=self._status))
            pre_target_x = (self.x1 + self.x2) // 2
            self._init = True
        phase = self._phase
        status = self._status

        # 0. 右エッジトレース→赤3000超でphase1へ
        if phase.get_phase() == 0:
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            _, _, red_pixel_count = find_bottle_center(image=image, color="red")
            if red_pixel_count > 3000:
                phase.next_phase()
                # phase1用 右モーター相対位置記録（絶対値）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 1. 赤ボトル中心追従（右モーター相対位置差分が1000未満の間、赤pixcelが500未満なら中央）
        if phase.get_phase() == 1:
            center, _, red_px = find_bottle_center(image=image, color="red")
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
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
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 2. 左エッジトレース（右モーター相対位置差分が1200未満の間）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 1200:
                left_x, _, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=300, threshold_value=80)
                target_x = left_x if left_x is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE1
            # 1200超えたら次フェーズへ
            phase.next_phase()
            # phase3用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 3. 左旋回（右モーター相対位置差分が390未満の間 左:0, 右:30）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 390:
                return None, (0, 30), Mode.CARRY_BOTTLE1
            # 390超えたら次フェーズへ
            phase.next_phase()
            # phase4用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 4. 直進（右モーター100ユニット移動まで）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 100:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            # 300超えたら次フェーズへ
            phase.next_phase()
            # phase5用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            pre_target_x = (self.x1 + self.x2) // 2

        # 5. 仮想ライン直進（右モーター1000ユニット移動まで, get_virtual_line_target_x, previous_center_x=pre_target_x）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 1000:
                # 右に障害物がある場合は左回避を明示
                temp_x = get_virtual_line_target_x(image, previous_center_x=pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    pre_target_x = temp_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    pre_target_x = target_x
                return target_x, None, Mode.CARRY_BOTTLE1
            # 1000超えたら次フェーズへ
            phase.next_phase()
            # phase6用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 6. 直進（右モーター1900ユニット移動まで）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 1900:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            # 1900超えたら次フェーズへ
            phase.next_phase()
            # phase7用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 7. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最大右モーター500ユニット、最低300ユニット旋回）
        if phase.get_phase() == 7:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_rotation_done = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_diff = abs(current_pos - position_start)
            minimum_rotation_done = position_diff >= 300
            position_limit_reached = position_diff >= 500
            
            # 最低300ユニット旋回後にblue_target検出を確認、500ユニット上限
            if minimum_rotation_done and blue_target_detected:
                phase.next_phase()
            elif not position_limit_reached:
                return None, (0, 30), Mode.CARRY_BOTTLE1
            else:
                # 500ユニット到達したが青が見つからない場合も次へ
                phase.next_phase()

        # 8. 青検出（1000超えたらphase9へ）
        if phase.get_phase() == 8:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                phase.next_phase()
            return target_x, None, Mode.CARRY_BOTTLE1

        # 9. 青1000以上の間center追従、500以下でphase10へ
        if phase.get_phase() == 9:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                phase.next_phase()
                # phase10用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            return target_x, None, Mode.CARRY_BOTTLE1

        # 10. 青500以下になってから右モーター300ユニット移動まで center追従、その後BACK_AND_TURN1
        if phase.get_phase() == 10:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            # 右モーター位置差分で継続判定
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 300:
                return target_x, None, Mode.CARRY_BOTTLE1
            else:
                phase.next_phase()
            
        if phase.get_phase() == 11:
            self._init = False
            return None, None, Mode.BACK_AND_TURN1

    def back_and_turn1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        back_and_turn1の位置判定バージョン。
        フェーズ:
        0. 後退（右モーター600ユニット移動まで, 両輪BASE_SPEED）
        1. 左旋回（右モーター450ユニット以上は必ず旋回、450超えた後is_x320_on_red_target(image, x_tolerance=60)検出または950ユニット到達まで左:0,右:30）
        2. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        """
        # 初回呼び出し時のみ初期化
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            # 初期位置記録を初回初期化時に実施
            self._phase.set_position_start("position_start", self.get_motor_position('right', status=self._status))
            self._init = True
        phase = self._phase
        status = self._status

        # 0. 後退（右モーター600ユニット移動まで）
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 600:
                return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN1
            phase.next_phase()
            # phase1用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 1. 左旋回（is_x320_on_red_target(image, x_tolerance=50)検出まで、最低右モーター450ユニット、最大右モーター950ユニット, 左:0, 右:30）
        if phase.get_phase() == 1:
            red_target_detected = is_x320_on_red_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_position_reached = False
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 450
            position_limit_reached = position_diff >= 940
        
            # 最低450ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (0, 30), Mode.BACK_AND_TURN1
            # 450ユニット超えてから、ターゲット検出または940ユニット到達まで継続
            if (not red_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.BACK_AND_TURN1
            phase.next_phase()

        # 2. 終了: 状態リセット
        if phase.get_phase() == 2:
            self._init = False
            return None, None, Mode.CARRY_BOTTLE2

    def carry_bottle2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:

        # 初回呼び出し時のみ初期化
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            # 初期位置記録を初回初期化時に実施
            self._phase.set_position_start("position_start", self.get_motor_position('right', status=self._status))
            pre_target_x = (self.x1 + self.x2) // 2
            self._init = True
        phase = self._phase
        status = self._status

        # phase 0: 赤ターゲット中心追従（青ピクセル数20000未満の間は赤中心追従、20000以上でphase1へ）
        if phase.get_phase() == 0:
            _, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count < 20000:
                # 赤ターゲット中心追従
                red_center_x = get_red_target_center_x(image)
                target_x = red_center_x if red_center_x is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                phase.next_phase()

        # phase 1: 青ボトル中心追従（青ピクセル数2000以上の間center追従、2000未満でphase2へ）
        if phase.get_phase() == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
            if blue_pixel_count >= 2000:
                # 青ボトル中心x座標へ追従
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                # 青ピクセル数が2000未満になった瞬間phase2へ
                phase.next_phase()
                # phase2用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 2: 右モーター200ユニット移動までcenter追従。200超えたらphase3へ
        if phase.get_phase() == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 200:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase3用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 3: 左黒ライン検出まで左旋回。最大右モーター1000ユニット
        if phase.get_phase() == 3:
            line_detected = is_left_black_line_detected(image)
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_limit_reached = abs(current_pos - position_start) >= 1000

            if (not line_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase4用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 4: 直進（右モーター870ユニット移動まで、両輪BASE_SPEED）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 870:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase5用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 5: 左旋回（右モーター380ユニット移動まで、左:0,右:30）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 380:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase6用 右モーター相対位置記録（絶対値）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 6: 直進（右モーター200ユニット移動まで、両輪BASE_SPEED）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 200:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase7用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            pre_target_x = (self.x1 + self.x2) // 2

        # phase 7: 仮想ライン直進（右モーター800ユニット移動まで、get_virtual_line_target_xで中心追従）
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 800:
                temp_x = get_virtual_line_target_x(image, previous_center_x=pre_target_x)
                if temp_x is not None:
                    target_x = temp_x
                    pre_target_x = temp_x
                else:
                    target_x = (self.x1 + self.x2) // 2
                    pre_target_x = target_x
                return target_x, None, Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase8用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 8: 直進（右モーター900ユニット移動まで、両輪BASE_SPEED）
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 900:
                return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase9用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 9: 左旋回（青ターゲット検出まで、最低右モーター300、最大500ユニット、左:0,右:30）
        if phase.get_phase() == 9:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_position_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 300
            position_limit_reached = position_diff >= 500
            # 最低300ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            # 300ユニット超えてから、ターゲット検出または500ユニット到達まで継続
            if (not blue_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            phase.next_phase()
            # phase10用 右モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # phase 10: 青検出（青ピクセル数1000超えたら次フェーズ、最大右モーター400ユニット）
        if phase.get_phase() == 10:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_limit_reached = abs(current_pos - position_start) >= 400

            if blue_pixel_count > 1000 or position_limit_reached:
                phase.next_phase()
                # phase11用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            return target_x, None, Mode.CARRY_BOTTLE2

        # phase 11: 青ピクセルが500以下まで減るまでcenter追従（500以下で次フェーズ、最大右モーター400ユニット）
        if phase.get_phase() == 11:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_limit_reached = abs(current_pos - position_start) >= 400

            if blue_pixel_count <= 500 or position_limit_reached:
                phase.next_phase()
                # phase12用 右モーター相対位置記録（get_motor_positionで統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            return target_x, None, Mode.CARRY_BOTTLE2

        # phase 12: 右モーター300ユニット移動までcenter追従、その後BACK_AND_TURN2へ遷移
        if phase.get_phase() == 12:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            if abs(current_pos - position_start) < 300:
                if center is not None:
                    target_x = center[0]
                else:
                    target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2            
            phase.next_phase()

        if phase.get_phase() == 13:
            self._init = False
            return None, None, Mode.BACK_AND_TURN2

    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        back_and_turn2の位置判定バージョン（右旋回のため左モーター位置追跡）。
        以下の順で動作する:
        0. 左モーター570ユニット移動まで後退（両輪BASE_SPEED）
        1. 左モーター370ユニット移動まで右旋回（左:30, 右:0）
        2. 終了後HEAD_GOALへ遷移（状態リセット）
        """

        # 初回呼び出し時のみ初期化
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            # 初期位置記録を初回初期化時に実施
            self._phase.set_position_start("position_start", self.get_motor_position('left', status=self._status))
            self._init = True
        phase = self._phase
        status = self._status

        # 0. 左モーター570ユニット移動まで後退
        if phase.get_phase() == 0:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('left', status=status)
            if abs(current_pos - position_start) < 570:
                return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN2
            phase.next_phase()
            # phase1用 左モーター相対位置記録（get_motor_positionで統一）
            phase.set_position_start("position_start", self.get_motor_position('left', status=status))

        # 1. 左モーター200～400ユニット移動まで右旋回（左:30, 右:0）
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('left', status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 200
            position_limit_reached = position_diff >= 400
            # 水平ライン検出→一般的な水平黒ライン検出に変更
            horizontal_line_detected = is_general_horizontal_line_detected(image)
        
            # 最低200ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (30, 0), Mode.BACK_AND_TURN2
            # 200ユニット超えてから、水平ライン検出または400ユニット到達まで継続
            if (not horizontal_line_detected) and (not position_limit_reached):
                return None, (30, 0), Mode.BACK_AND_TURN2
            # 条件を満たしたので次のフェーズへ
            phase.next_phase()

        # 2. 終了: 状態リセット
        if phase.get_phase() == 2:
            self._init = False
            return None, None, Mode.HEAD_GOAL

    def heading_goal_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        heading_goalの位置判定バージョン（右モーターBの相対位置追跡）。
        フェーズ:
        0. intersection_y=450で黒水平ライン検出まで中央追従（右モーター距離制限なし）
            - 黒水平ライン検出でphase1へ。右モーター位置記録。
        1. 右モーターBの移動距離が300未満なら中央追従、300以上でphase2へ遷移
            - 300到達で右モーター位置記録。
        2. 左旋回（右モーターBの移動距離100未満は常に左旋回。100以上で垂直黒ライン検出開始。600未満の間は左:0,右:30で継続。垂直ライン検出または600到達でphase3へ）
            - 100未満はis_vertical_black_line_detected呼ばない。100以上で検出開始。
        3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央）
            - 青ライン検出時に右モーター位置記録。
        4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        各行コメントも実装内容と完全一致させること。
        """

        # 初回呼び出し時のみ初期化
        if not self._init:
            self._phase = PhaseManager()
            self._status = self.get_motor_position(mode="status")
            # 初期位置記録を初回初期化時に実施
            self._phase.set_position_start("position_start", self.get_motor_position('right', status=self._status))
            self._init = True
        phase = self._phase
        status = self._status

        # 0. intersection_y=450で黒水平ライン検出まで中央追従（距離制限なし）
        if phase.get_phase() == 0:
            # intersection_y=450で黒水平ライン検出
            if is_horizontal_black_line_detected(image, intersection_y=450):
                phase.next_phase()
                # phase1用 右モーター相対位置記録（get_motor_positionで完全統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            else:
                # ライン未検出時は中央追従
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 1. 右モーターBの移動距離が300未満なら中央追従、300以上でphase2へ遷移
        if phase.get_phase() == 1:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_diff = abs(current_pos - position_start)
            target_x = (self.x1 + self.x2) // 2
            if position_diff < 300:
                # 300未満は中央追従
                return target_x, None, Mode.HEAD_GOAL
            else:
                phase.next_phase()  # phase2へ遷移
                # phase2用 右モーター相対位置記録（get_motor_positionで完全統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))

        # 2. 左旋回（右モーターBの移動距離100未満は常に左旋回。100以上で垂直黒ライン検出開始。600未満の間は左:0,右:30で継続。垂直ライン検出または600到達でphase3へ）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            # get_motor_positionで完全統一
            current_pos = self.get_motor_position('right', status=status)
            position_diff = abs(current_pos - position_start)
            minimum_position_reached = position_diff >= 100  # 100未満はライン検出しない
            position_limit_reached = position_diff >= 600
            if not minimum_position_reached:
                # 100未満はis_vertical_black_line_detected呼ばない
                return None, (0, 30), Mode.HEAD_GOAL
            # 100以上で垂直ライン検出
            vertical_line_detected = is_vertical_black_line_detected(image)
            if (not vertical_line_detected) and (not position_limit_reached):
                # 垂直ライン未検出・600未満は左旋回継続
                return None, (0, 30), Mode.HEAD_GOAL
            # 垂直ライン検出または600到達でphase3へ
            phase.next_phase()  # phase3へ遷移

        # 3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央）
        if phase.get_phase() == 3:
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if left_x is not None:
                target_x = left_x
            else:
                target_x = (self.x1 + self.x2) // 2
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                phase.next_phase()  # phase4へ遷移
                # phase4用 右モーター相対位置記録（get_motor_positionで完全統一）
                phase.set_position_start("position_start", self.get_motor_position('right', status=status))
            # 青ライン未検出時も左エッジ追従
            return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        if phase.get_phase() == 4:
            position_limit_reached = False
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position('right', status=status)
            position_limit_reached = abs(current_pos - position_start) >= 600
            if position_limit_reached:
                phase.next_phase()
            # 600未満は左エッジ追従
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if left_x is not None:
                target_x = left_x
            else:
                target_x = (self.x1 + self.x2) // 2
            return target_x, None, Mode.HEAD_GOAL

        if phase.get_phase() == 5:
            # 600到達で状態リセットしPAUSE
            self._init = False
            return None, None, Mode.PAUSE