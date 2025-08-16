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
        self._state = {}  # 各アクションの状態管理dict
        self.x1, self.y1, self.x2, self.y2 = ROI_CNN  # ROI座標

    def get_motor_position(self, side: str = "right", mode: str = "position", status=None):
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
                return None
            return status
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
        ・青ピクセル数が3000を超えるまでは青重心追従（target_xに重心x座標、速度指令はNone）
        ・一度3000超えた後、3000未満になったら0.8秒間直進（左右45）
        ・0.8秒経過後にPAUSEへ遷移
        戻り値: (target_x, (左速度, 右速度), モード)
        """
        state = self._state.setdefault("blue_bottle_catch", {"detected": False, "below3000_time": None})
        blue_cx, _, blue_pixel_count = find_bottle_center(image, color="blue")
        now = time.time()
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        BASE_SPEED = 45  # run_manual.pyと合わせる
        target_x = None
        left_speed = right_speed = None
        mode = Mode.BLUE_BOTTLE_CATCH
        if not state["detected"]:
            if blue_pixel_count > 3000:
                state["detected"] = True
            state["below3000_time"] = None
            if blue_cx is not None:
                target_x = blue_cx[0]
            else:
                target_x = (x1 + x2) // 2
            return target_x, None, mode
        else:
            if blue_pixel_count < 3000:
                if state.get("below3000_time") is None:
                    state["below3000_time"] = now
                # 0.8秒間直進
                if now - state["below3000_time"] < 0.8:
                    target_x = (x1 + x2) // 2
                    left_speed = right_speed = BASE_SPEED
                    return target_x, (left_speed, right_speed), mode
                else:
                    # 状態リセットしPAUSEへ
                    state["detected"] = False
                    state["below3000_time"] = None
                    target_x = (x1 + x2) // 2
                    return target_x, (0, 0), Mode.PAUSE
            else:
                state["below3000_time"] = None
                if blue_cx is not None:
                    target_x = blue_cx[0]
                else:
                    target_x = (x1 + x2) // 2
                return target_x, None, mode


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
        state = self._state.setdefault("avoid_obstacle_relative", {
            "phase": 0,
            "right_position_start": None,
        })
        status = self.get_motor_position(mode="status")

        # 0. 左旋回（右モーター500ユニット移動まで, 左:40, 右:70）
        if state["phase"] == 0:
            if state["right_position_start"] is None:
                state["right_position_start"] = self.get_motor_position('right', status=status)

            current_pos = self.get_motor_position('right', status=status)
            if state["right_position_start"] is not None:
                if abs(current_pos - state["right_position_start"]) < 500:
                    return None, (40, 70), Mode.AVOID_OBSTACLE
                else:
                    # 500ユニット到達したので次のフェーズへ
                    state["phase"] = 1
                    # phase1用も右モーター相対位置記録（絶対値）
                    state["right_position_start"] = self.get_motor_position('right', status=status)
                # ステータス取得失敗時は継続
                return None, (40, 70), Mode.AVOID_OBSTACLE

        # 1. 右旋回（右モーター700ユニット移動まで, 左:80, 右:50）
        if state["phase"] == 1:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                current_pos = abs(status.motors["B"].relative_position) if status.motors["B"].relative_position is not None else 0
                if abs(current_pos - state["right_position_start"]) < 550:
                    return None, (70, 40), Mode.AVOID_OBSTACLE
                else:
                    # 650ユニット到達したので次のフェーズへ
                    state["phase"] = 2
                    # phase2用も右モーター相対位置記録（絶対値）
                    state["right_position_start"] = self.get_motor_position('right', status=status)
            else:
                # ステータス取得失敗時は継続
                return None, (70, 40), Mode.AVOID_OBSTACLE

        # 2. 左旋回（右モーター350ユニット移動まで, 左:40, 右:70）
        if state["phase"] == 2:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                distance = abs(current_pos - state["right_position_start"])
                # 走行距離が200未満なら常に(40,70)で走行
                if distance < 250:
                    return None, (40, 70), Mode.AVOID_OBSTACLE
                # 250以上350未満の間はis_vertical_black_line_detected(image)がTrueなら即フェーズ3へ
                elif distance < 350:
                    if is_vertical_black_line_detected(image):
                        state["phase"] = 3
                        # すぐ次の処理でphase3に入る
                    else:
                        return None, (40, 70), Mode.AVOID_OBSTACLE
                # 450以上なら強制的にフェーズ3へ
                else:
                    state["phase"] = 3
            else:
                # ステータス取得失敗時は継続
                return None, (40, 70), Mode.AVOID_OBSTACLE

        # 3. チェーン終了でリセット
        if state["phase"] == 3:
            self._state["avoid_obstacle_relative"] = {
                "phase": 0, 
                "right_position_start": None,
            }
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
        state = self._state.setdefault("carry_bottle1_relative", {
            "phase": 0,
            "pre_target_x": None,
            "right_position_start": None,
        })
        status = self.get_motor_position(mode="status")

        # 0. 右エッジトレース→赤3000超でphase1へ
        if state["phase"] == 0:
            _, right_x, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            target_x = right_x if right_x is not None else (self.x1 + self.x2) // 2
            _, _, red_pixel_count = find_bottle_center(image=image, color="red")
            if red_pixel_count > 3000:
                state["phase"] = 1
                # phase1用 右モーター相対位置記録（絶対値）
                state["right_position_start"] = self.get_motor_position('right', status=status)
            else:
                return target_x, None, Mode.CARRY_BOTTLE1

        # 1. 赤ボトル中心追従（右モーター相対位置差分が1000未満の間、赤pixcelが500未満なら中央）
        if state["phase"] == 1:
            center, _, red_px = find_bottle_center(image=image, color="red")
            # None チェックは不要 - find_bottle_center は常に3つの値を返す（centerがNoneの場合もあるが）
            # 右モーター相対位置差分で継続判定
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 1000:
                    if center is not None and red_px is not None and red_px >= 500:
                        target_x = center[0]
                    else:
                        target_x = (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 1000超えたら次フェーズへ
            state["phase"] = 2
            # phase2用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 2. 左エッジトレース（右モーター相対位置差分が1200未満の間）
        if state["phase"] == 2:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 1200:
                    left_x, _, _ = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=300, threshold_value=80)
                    target_x = left_x if left_x is not None else (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 1200超えたら次フェーズへ
            state["phase"] = 3
            state["pre_target_x"] = (self.x1 + self.x2) // 2
            # phase3用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 3. 左旋回（右モーター相対位置差分が390未満の間 左:0, 右:30）
        if state["phase"] == 3:
            if state["right_position_start"] is None:
                state["right_position_start"] = self.get_motor_position('right', status=status)
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 390:
                    return None, (0, 30), Mode.CARRY_BOTTLE1
            # 390超えたら次フェーズへ
            state["phase"] = 4
            # phase4用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 4. 直進（右モーター100ユニット移動まで）
        if state["phase"] == 4:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 100:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            # 300超えたら次フェーズへ
            state["phase"] = 5
            state["pre_target_x"] = (self.x1 + self.x2) // 2
            # phase5用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 5. 仮想ライン直進（右モーター1000ユニット移動まで, get_virtual_line_target_x, previous_center_x=pre_target_x）
        if state["phase"] == 5:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 1000:
                    pre_target_x = state.get("pre_target_x")
                    # 右に障害物がある場合は左回避を明示
                    temp_x = get_virtual_line_target_x(image, previous_center_x=pre_target_x)
                    if temp_x is not None:
                        target_x = temp_x
                        state["pre_target_x"] = temp_x
                    elif pre_target_x is not None:
                        target_x = pre_target_x
                    else:
                        target_x = (self.x1 + self.x2) // 2
                        state["pre_target_x"] = target_x
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 1000超えたら次フェーズへ
            state["phase"] = 6
            # phase6用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 6. 直進（右モーター1900ユニット移動まで）
        if state["phase"] == 6:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 1900:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE1
            # 1900超えたら次フェーズへ
            state["phase"] = 7
            # phase7用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 7. 左旋回（is_x320_on_blue_targetがTrueになるまで左:0, 右:30で旋回、最大右モーター500ユニット、最低300ユニット旋回）
        if state["phase"] == 7:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_rotation_done = False
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                position_diff = abs(current_pos - state["right_position_start"])
                minimum_rotation_done = position_diff >= 300
                position_limit_reached = position_diff >= 500
            
            # 最低300ユニット旋回後にblue_target検出を確認、500ユニット上限
            if minimum_rotation_done and blue_target_detected:
                state["phase"] = 8
            elif not position_limit_reached:
                return None, (0, 30), Mode.CARRY_BOTTLE1
            else:
                # 500ユニット到達したが青が見つからない場合も次へ
                state["phase"] = 8

        # 8. 青検出（1000超えたらphase9へ）
        if state["phase"] == 8:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count > 1000:
                state["phase"] = 9
            return target_x, None, Mode.CARRY_BOTTLE1

        # 9. 青1000以上の間center追従、500以下でphase10へ
        if state["phase"] == 9:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            if blue_pixel_count <= 500:
                state["phase"] = 10
                # phase10用 右モーター相対位置記録（get_motor_positionで統一）
                state["right_position_start"] = self.get_motor_position('right', status=status)
            return target_x, None, Mode.CARRY_BOTTLE1

        # 10. 青500以下になってから右モーター300ユニット移動まで center追従、その後BACK_AND_TURN1
        if state["phase"] == 10:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            # 右モーター位置差分で継続判定
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 300:
                    return target_x, None, Mode.CARRY_BOTTLE1
            # 状態リセット
            self._state["carry_bottle1_relative"] = {
                "phase": 0, 
                "pre_target_x": None,
                "right_position_start": None,
            }
            return None, None, Mode.BACK_AND_TURN1

    def back_and_turn1_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        back_and_turn1の位置判定バージョン。
        フェーズ:
        0. 後退（右モーター600ユニット移動まで, 両輪BASE_SPEED）
        1. 左旋回（右モーター450ユニット以上は必ず旋回、450超えた後is_x320_on_red_target(image, x_tolerance=60)検出または950ユニット到達まで左:0,右:30）
        2. 終了: 状態リセットしCARRY_BOTTLE2へ遷移
        """
        state = self._state.setdefault("back_and_turn1_relative", {
            "phase": 0,
            "right_position_start": None,
        })
        status = self.get_motor_position(mode="status") 

        # 0. 後退（右モーター600ユニット移動まで）
        if state["phase"] == 0:
            if state["right_position_start"] is None:
                state["right_position_start"] = self.get_motor_position('right', status=status)
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 600:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN1
            state["phase"] = 1
            # phase1用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # 1. 左旋回（is_x320_on_red_target(image, x_tolerance=50)検出まで、最低右モーター450ユニット、最大右モーター950ユニット, 左:0, 右:30）
        if state["phase"] == 1:
            red_target_detected = is_x320_on_red_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_position_reached = False
            position_limit_reached = False
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                position_diff = abs(current_pos - state["right_position_start"])
                minimum_position_reached = position_diff >= 450
                position_limit_reached = position_diff >= 940
            
            # 最低450ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (0, 30), Mode.BACK_AND_TURN1
            # 450ユニット超えてから、ターゲット検出または940ユニット到達まで継続
            if (not red_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.BACK_AND_TURN1
            state["phase"] = 2

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn1_relative"] = {"phase": 0, "right_position_start": None}
            return None, None, Mode.CARRY_BOTTLE2

    def carry_bottle2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:

        state = self._state.setdefault("carry_bottle2_relative", {
            "phase": 0,
            "pre_target_x": None,
            "right_position_start": None,
        })
        status = self.get_motor_position(mode="status")

        # phase 0: 赤ターゲット中心追従（青ピクセル数20000未満の間は赤中心追従、20000以上でphase1へ）
        if state["phase"] == 0:
            _, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if blue_pixel_count < 20000:
                # 赤ターゲット中心追従
                red_center_x = get_red_target_center_x(image)
                target_x = red_center_x if red_center_x is not None else (self.x1 + self.x2) // 2
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                state["phase"] = 1

        # phase 1: 青ボトル中心追従（青ピクセル数2000以上の間center追従、2000未満でphase2へ）
        if state["phase"] == 1:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            target_x = center[0] if center is not None else (self.x1 + self.x2) // 2
            if blue_pixel_count >= 2000:
                # 青ボトル中心x座標へ追従
                return target_x, None, Mode.CARRY_BOTTLE2
            else:
                # 青ピクセル数が2000未満になった瞬間phase2へ
                state["phase"] = 2
                # phase2用 右モーター相対位置記録（get_motor_positionで統一）
                state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 2: 右モーター200ユニット移動までcenter追従。200超えたらphase3へ
        if state["phase"] == 2:
            center, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 200:
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE2
            state["phase"] = 3
            # phase3用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 3: 左黒ライン検出まで左旋回。最大右モーター1000ユニット
        if state["phase"] == 3:
            line_detected = is_left_black_line_detected(image)
            position_limit_reached = False
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 1000
            
            if (not line_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 4
            # phase4用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 4: 直進（右モーター870ユニット移動まで、両輪BASE_SPEED）
        if state["phase"] == 4:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 870:
                    state["pre_target_x"] = (self.x1 + self.x2) // 2
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 5
            # phase5用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 5: 左旋回（右モーター380ユニット移動まで、左:0,右:30）
        if state["phase"] == 5:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 380:
                    return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 6
            # phase6用 右モーター相対位置記録（絶対値）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 6: 直進（右モーター200ユニット移動まで、両輪BASE_SPEED）
        if state["phase"] == 6:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 200:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 7
            # phase7用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 7: 仮想ライン直進（右モーター800ユニット移動まで、get_virtual_line_target_xで中心追従）
        if state["phase"] == 7:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 800:
                    pre_target_x = state.get("pre_target_x")
                    temp_x = get_virtual_line_target_x(image, previous_center_x=pre_target_x)
                    if temp_x is not None:
                        target_x = temp_x
                        state["pre_target_x"] = temp_x
                    elif pre_target_x is not None:
                        target_x = pre_target_x
                    else:
                        target_x = (self.x1 + self.x2) // 2
                        state["pre_target_x"] = target_x
                    return target_x, None, Mode.CARRY_BOTTLE2
            state["phase"] = 8
            # phase8用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 8: 直進（右モーター900ユニット移動まで、両輪BASE_SPEED）
        if state["phase"] == 8:
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 900:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.CARRY_BOTTLE2
            state["phase"] = 9
            # phase9用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 9: 左旋回（青ターゲット検出まで、最低右モーター300、最大500ユニット、左:0,右:30）
        if state["phase"] == 9:
            blue_target_detected = is_x320_on_blue_target(image, x_tolerance=60)
            position_limit_reached = False
            minimum_position_reached = False
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                position_diff = abs(current_pos - state["right_position_start"])
                minimum_position_reached = position_diff >= 300
                position_limit_reached = position_diff >= 500
            # 最低300ユニットは必ず旋回
            if not minimum_position_reached:
                return None, (0, 30), Mode.CARRY_BOTTLE2
            # 300ユニット超えてから、ターゲット検出または500ユニット到達まで継続
            if (not blue_target_detected) and (not position_limit_reached):
                return None, (0, 30), Mode.CARRY_BOTTLE2
            state["phase"] = 10
            # phase10用 右モーター相対位置記録（get_motor_positionで統一）
            state["right_position_start"] = self.get_motor_position('right', status=status)

        # phase 10: 青検出（青ピクセル数1000超えたら次フェーズ、最大右モーター400ユニット）
        if state["phase"] == 10:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 400
            
            if blue_pixel_count > 1000 or position_limit_reached:
                state["phase"] = 11
                # phase11用 右モーター相対位置記録（get_motor_positionで統一）
                state["right_position_start"] = self.get_motor_position('right', status=status)
            return target_x, None, Mode.CARRY_BOTTLE2

        # phase 11: 青ピクセルが500以下まで減るまでcenter追従（500以下で次フェーズ、最大右モーター400ユニット）
        if state["phase"] == 11:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            if center is not None:
                target_x = center[0]
            else:
                target_x = (self.x1 + self.x2) // 2
            
            position_limit_reached = False
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 400
            
            if blue_pixel_count <= 500 or position_limit_reached:
                state["phase"] = 12
                # phase12用 右モーター相対位置記録（get_motor_positionで統一）
                state["right_position_start"] = self.get_motor_position('right', status=status)
            return target_x, None, Mode.CARRY_BOTTLE2

        # phase 12: 右モーター300ユニット移動までcenter追従、その後BACK_AND_TURN2へ遷移
        if state["phase"] == 12:
            center, _, blue_pixel_count = find_blue_target_center(image, gray_ellipse_enable=False)
            
            if state["right_position_start"] is not None:
                current_pos = self.get_motor_position('right', status=status)
                if abs(current_pos - state["right_position_start"]) < 300:
                    if center is not None:
                        target_x = center[0]
                    else:
                        target_x = (self.x1 + self.x2) // 2
                    return target_x, None, Mode.CARRY_BOTTLE2
            # 状態リセット
            self._state["carry_bottle2_relative"] = {
                "phase": 0, 
                "pre_target_x": None,
                "right_position_start": None,
            }
            return None, None, Mode.BACK_AND_TURN2

    def back_and_turn2_relative(self, image: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[int, int]], Mode]:
        """
        back_and_turn2の位置判定バージョン（右旋回のため左モーター位置追跡）。
        以下の順で動作する:
        0. 左モーター570ユニット移動まで後退（両輪BASE_SPEED）
        1. 左モーター370ユニット移動まで右旋回（左:30, 右:0）
        2. 終了後HEAD_GOALへ遷移（状態リセット）
        """
        state = self._state.setdefault("back_and_turn2_relative", {
            "phase": 0,
            "left_position_start": None,
        })
        status = self.get_motor_position(mode="status")

        # 0. 左モーター570ユニット移動まで後退
        if state["phase"] == 0:
            if state["left_position_start"] is None:
                state["left_position_start"] = self.get_motor_position('left', status=status)

            if state["left_position_start"] is not None:
                current_pos = self.get_motor_position('left', status=status)
                if abs(current_pos - state["left_position_start"]) < 570:
                    return None, (BASE_SPEED, BASE_SPEED), Mode.BACK_AND_TURN2
            state["phase"] = 1
            # phase1用 左モーター相対位置記録（get_motor_positionで統一）
            state["left_position_start"] = self.get_motor_position('left', status=status)

        # 1. 左モーター200～400ユニット移動まで右旋回（左:30, 右:0）
        if state["phase"] == 1:
            if state["left_position_start"] is not None:
                current_pos = self.get_motor_position('left', status=status)
                position_diff = abs(current_pos - state["left_position_start"])
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
                state["phase"] = 2
            else:
                # ステータス取得失敗時は継続
                return None, (30, 0), Mode.BACK_AND_TURN2

        # 2. 終了: 状態リセット
        if state["phase"] == 2:
            self._state["back_and_turn2_relative"] = {"phase": 0, "left_position_start": None}
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
        state = self._state.setdefault("heading_goal_relative", {
            "phase": 0,  # 現在のフェーズ
            "right_position_start": None,  # 右モーターBの基準位置
        })
        status = self.et.get_spike_status()  # ロボットの現在ステータス取得

        # 0. intersection_y=450で黒水平ライン検出まで中央追従（距離制限なし）
        if state["phase"] == 0:
            # 右モーター基準位置未設定なら記録
            if state["right_position_start"] is None:
                if status is not None and status.motors.get("B") is not None and status.motors["B"].relative_position is not None:
                    state["right_position_start"] = abs(status.motors["B"].relative_position) if status.motors["B"].relative_position is not None else 0
                else:
                    state["right_position_start"] = None
            # intersection_y=450で黒水平ライン検出
            if is_horizontal_black_line_detected(image, intersection_y=450):
                state["phase"] = 1  # phase1へ遷移
                # phase1用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    right_position = status.motors["B"].relative_position
                    state["right_position_start"] = abs(right_position) if right_position is not None else None
                else:
                    state["right_position_start"] = None
            else:
                # ライン未検出時は中央追従
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 1. 右モーターBの移動距離が300未満なら中央追従、300以上でphase2へ遷移
        if state["phase"] == 1:
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                right_position = status.motors["B"].relative_position
                current_pos = abs(right_position) if right_position is not None else 0
                position_diff = abs(current_pos - state["right_position_start"])
                target_x = (self.x1 + self.x2) // 2
                if position_diff < 300:
                    # 300未満は中央追従
                    return target_x, None, Mode.HEAD_GOAL
                else:
                    state["phase"] = 2  # phase2へ遷移
                    # phase2用 右モーター相対位置記録（絶対値）
                    if status is not None and status.motors.get("B") is not None:
                        rel_pos = status.motors["B"].relative_position
                        state["right_position_start"] = abs(rel_pos) if rel_pos is not None else None
                    else:
                        state["right_position_start"] = None
            else:
                # ステータス取得失敗時も中央追従
                target_x = (self.x1 + self.x2) // 2
                return target_x, None, Mode.HEAD_GOAL

        # 2. 左旋回（右モーターBの移動距離100未満は常に左旋回。100以上で垂直黒ライン検出開始。600未満の間は左:0,右:30で継続。垂直ライン検出または600到達でphase3へ）
        if state["phase"] == 2:
            if status is not None and status.motors.get("B") is not None:
                rel_pos = status.motors["B"].relative_position
                current_pos = abs(rel_pos) if rel_pos is not None else 0
                position_diff = abs(current_pos - state["right_position_start"]) if state["right_position_start"] is not None else 0
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
                state["phase"] = 3
            else:
                # ステータス取得失敗時は左旋回継続
                return None, (0, 30), Mode.HEAD_GOAL

        # 3. 左エッジトレース（青ライン検出でphase4へ。左エッジがなければ中央）
        if state["phase"] == 3:
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if left_x is not None:
                target_x = left_x
            else:
                target_x = (self.x1 + self.x2) // 2
            blue_line = get_is_blue_line_at_y(image, target_y=OFFSET_Y)
            if blue_line:
                state["phase"] = 4  # phase4へ遷移
                # phase4用 右モーター相対位置記録（絶対値）
                if status is not None and status.motors.get("B") is not None:
                    right_position = status.motors["B"].relative_position
                    state["right_position_start"] = abs(right_position) if right_position is not None else None
                else:
                    state["right_position_start"] = None
                return target_x, None, Mode.HEAD_GOAL
            # 青ライン未検出時も左エッジ追従
            return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        if state["phase"] == 4:
            position_limit_reached = False
            if status is not None and status.motors.get("B") is not None and state["right_position_start"] is not None:
                right_position = status.motors["B"].relative_position
                current_pos = abs(right_position) if right_position is not None else 0
                position_limit_reached = abs(current_pos - state["right_position_start"]) >= 600
            if position_limit_reached:
                # 600到達で状態リセットしPAUSE
                self._state["heading_goal_relative"] = {"phase": 0, "right_position_start": None}
                return None, None, Mode.PAUSE
            # 600未満は左エッジ追従
            left_x, right_x, mask = get_line_edges_at_y(image=image, roi=ROI_CNN, target_y=OFFSET_Y, threshold_value=80)
            if left_x is not None:
                target_x = left_x
            else:
                target_x = (self.x1 + self.x2) // 2
            return target_x, None, Mode.HEAD_GOAL