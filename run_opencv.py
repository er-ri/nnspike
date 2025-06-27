#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

このスクリプトは、OpenCV画像処理によるライン追従ロボット制御を行います。
ニューラルネットワークは使わず、カメラ画像からラインの重心を検出し、PID制御で走行します。

【速度調整パラメータ】
- BASE_POWER: 直線時の基準パワー
- TURN_REDUCTION_FACTOR: カーブ時の速度低減率（0.0-1.0）
- TURN_THRESHOLD: カーブ判定のピクセル閾値

【PID調整パラメータ】
- Kp, Ki, Kd: ステアリング補正用のPIDパラメータ

【全体設計方針】
- main関数は「全体の流れ・状態遷移・例外処理・リソース管理」のみを記述し、
  各種ハードウェア操作や状態取得はActionManager等の専用クラスに集約。
- mainループの各処理段階（画像取得→センサー取得→画像処理→制御→可視化→送信/保存→周期調整）を明確にコメントで区切る。
- 例外発生時も安全停止・リソース解放を徹底。
- OOP設計・責務分担を明確化し、拡張性・保守性を重視。
"""

# ==== ユーザー調整用パラメータ（ここだけ編集すればOK） ====
# ROI_OPENCV: OpenCV画像処理で使用する領域（左上x, 左上y, 右下x, 右下y）
ROI_OPENCV = (180, 300, 460, 400)  # 左右をそれぞれ30pxずつ内側に狭めた例
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480                 # カメラ画像の高さ
BASE_POWER = 45                    # カーブ時の基準パワー 40→45にUP
STRAIGHT_POWER = 60                # 直線時の推奨パワー 50→60にUP
CURVE_POWER = 35                   # 急カーブ時の最低パワー 30→35にUP
CURVE_THRESHOLD_DEG = 10           # カーブ判定閾値（度数法, SENSITIVITY=1.0時の推奨値）
STRAIGHT_THRESHOLD_DEG = 3         # 直線判定のしきい値（ユーザー調整用, デフォルト3度, STRAIGHT_THRESHOLD_DEGで指定）
SENSITIVITY = 1.0                  # ピクセル→theta変換感度
MAX_POWER_DIFF = 45       # 最大旋回時の左右パワー差（%） 45→55にUP
MAX_THETA_DEG = 55        # 最大旋回角（度数法, 55→65度にUP）
# 黒判定の閾値（反射光R: 40以下, color: 150以下なら黒と判定）
BLACK_REFLECTED_THRESHOLD = 40
BLACK_COLOR_THRESHOLD = 150
# 青判定の閾値（色値: 30以下, 反射光: 60以下なら青と判定）
BLUE_COLOR_THRESHOLD = 30
BLUE_REFLECTED_THRESHOLD = 60
# ================================================

import cv2
import math
import time
import socket
import pickle
import struct
import argparse
import numpy as np
from nnspike.unit import ETRobot
from nnspike.utils.control import ControlCalculator
from nnspike.utils import (
    draw_driving_info,
    SensorRecorder,
    PIDController,
)
from enum import Enum, auto

# --- ソケット通信設定 ---
HOST_IP_ADDRESS = (
    "192.168.137.1"  # Raspberry PiがPCへ送信する宛先IP
)

# --- カメラ初期化 ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


# AsyncSensorReader クラスを削除 - メインループで直接センサー値を取得するように変更


class Mode(Enum):
    LINE_TRACE = auto()         # 通常のライン走行
    DIST_STOP = auto()          # 超音波距離停止モード（障害物検知後の一時停止・距離再確認）
    OBSTACLE_AVOID = auto()     # 障害物回避動作（オブスタクルボトル回避）
    SMART_CARRY_1 = auto()      # スマートキャリー1回目
    SMART_CARRY_2 = auto()      # スマートキャリー2回目
    GOAL = auto()               # ゴール到達モード（ゴールに向かう処理とゴール停止をこのモードで実装する構想）

class ModeManager:
    """
    動作モードの状態遷移を管理するクラス。
    LINE_TRACE: 通常のライン走行
    DIST_STOP: 超音波距離停止モード（障害物検知後の一時停止・距離再確認）
    OBSTACLE_AVOID: 障害物回避動作（オブスタクルボトル回避）
    SMART_CARRY_1: スマートキャリー1回目
    SMART_CARRY_2: スマートキャリー2回目
    GOAL: ゴール到達モード（ゴールに向かう処理とゴール停止をこのモードで実装する構想）

    # MEMO:
    # スマートキャリーモード(SMART_CARRY_1, SMART_CARRY_2)やGOALモードも、
    # 超音波センサーやobject_detected（物体判定結果）、relative_position（モーター相対位置）等を組み合わせて遷移判定する構想。
    # 例: object_detected==2（交差点やゴール判定値）でGOALモードへ遷移。
    # 必要に応じてupdate()内でこれらの値を参照し、より柔軟なモード遷移を実装すること。
    """
    def __init__(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None
        # theta, pid_corrected_theta, current_powerはActionManagerに移動
    def update(self, distance):
        # --- 障害物検知・停止用パラメータをローカル変数で定義 ---
        OBSTACLE_DETECT_DISTANCE = 50  # 障害物検知のしきい値[cm]
        DIST_STOP_DURATION = 2.0       # 距離停止モードの待機時間[秒]
        # 距離センサー値に応じてモード遷移
        if self.mode == Mode.LINE_TRACE:
            if distance is not None and distance < OBSTACLE_DETECT_DISTANCE:
                self.mode = Mode.DIST_STOP
                self.obstacle_detected_time = time.time()
        elif self.mode == Mode.DIST_STOP:
            if distance is not None and distance >= OBSTACLE_DETECT_DISTANCE * (50/30):
                self.mode = Mode.LINE_TRACE
                self.obstacle_detected_time = None
            elif time.time() - self.obstacle_detected_time >= DIST_STOP_DURATION:
                self.mode = Mode.OBSTACLE_AVOID
        elif self.mode == Mode.OBSTACLE_AVOID:
            pass
        # SMART_CARRY_1, SMART_CARRY_2, GOALへの遷移は必要に応じて追加
    def reset(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None

    def update_and_act(self, action_manager, offset_pixels=None, calc=None):
        self.update(action_manager.distance)
        if self.mode == Mode.LINE_TRACE:
            action_manager.do_line_trace(offset_pixels, calc)
        elif self.mode == Mode.DIST_STOP:
            action_manager.do_dist_stop()
        elif self.mode == Mode.OBSTACLE_AVOID:
            action_manager.do_obstacle_avoid()
            if action_manager.is_finished():
                print(f"[DEBUG] OBSTACLE_AVOID終了: state={action_manager.state}, finished={action_manager.finished}, 時刻={time.strftime('%H:%M:%S')}")
                self.mode = Mode.LINE_TRACE  # state3後に必ずLINE_TRACEへ遷移
                self.obstacle_detected_time = None
                action_manager.reset()  # 回避動作の状態もリセット
        elif self.mode == Mode.GOAL:
            pass
        else:
            pass

# --- 固有動作管理クラス（回避・今後の特殊動作用） ---
class ActionManager:
    """
    障害物回避やゴール到達時などの固有動作を管理する拡張用クラス。
    例: 45度左回転→右弧旋回→45度左回転で回避（非ブロッキングで各動作を進める）

    # MEMO:
    # 今後、オブスタクルボトルの回避だけでなく、
    # スマートキャリーモード(SMART_CARRY_1, SMART_CARRY_2)でキャリーボトルの運搬・制御、
    # GOALモード時の固有動作（例: 停止、アーム動作、サウンド再生等）も追加予定。
    # 必要に応じてstateやobstacle_avoid_step()の分岐・処理を拡張すること。
    """
    def __init__(self, et=None):
        if et is None:
            self.et = ETRobot()
        else:
            self.et = et
        self.state = 0
        self._reset_action_vars()
        self.pid = PIDController(
            Kp=1.5,
            Ki=0,
            Kd=0.4,
            setpoint=0,
            output_limits=(-0.25, 0.25),
        )
        self.et.set_motor_relative_position(left_position=0, right_position=0)
        self.reset_control_values()  # theta, pid_corrected_theta, current_powerをまとめてリセット

    def reset(self):
        self.state = 0
        self._reset_action_vars()

    def _reset_action_vars(self):
        self.action_sent = False
        self.start_time = None
        self.finished = False
        self.left_power = 0
        self.right_power = 0

    def is_finished(self):
        return self.finished

    def apply_power(self):
        """現在のleft_power, right_powerをロボットに反映"""
        self.et.set_motor_forward_power(left_power=self.left_power, right_power=self.right_power)

    def reset_control_values(self):
        self.theta = 0
        self.pid_corrected_theta = 0
        self.current_power = 0

    def do_line_trace(self, offset_pixels, calc):
        # --- ライントレース時の進行角度・推奨速度・PID補正・左右パワー計算 ---
        # MAX_THETA_DEG: 最大旋回角（度数法, ユーザー調整パラメータで一元管理）
        # MAX_POWER_DIFF: 最大旋回時の左右パワー差（%）, ユーザー調整パラメータで一元管理
        # offset_pixels: ライン重心のオフセット（ピクセル単位, 画像中心からのズレ）
        # calc: ControlCalculatorインスタンス（theta計算や速度調整ロジックを内包）
        theta = calc.calculate_theta_from_pixels(offset_pixels)  # オフセットピクセル→進行角度（ラジアン）へ変換
        current_power = calc.calculate_adaptive_speed(abs(theta))  # 進行角度に応じて推奨速度（パワー）を自動調整
        pid_corrected_theta = self.pid.update(theta)  # PID制御で進行角度を補正
        max_theta = math.radians(MAX_THETA_DEG)  # 最大旋回角をラジアンに変換（ユーザー調整パラメータを参照）
        power_adjustment = int((pid_corrected_theta / max_theta) * MAX_POWER_DIFF)  # PID補正値をパワー差分に変換
        self.left_power = int(current_power - power_adjustment)   # 左右パワーを計算
        self.right_power = int(current_power + power_adjustment)
        self.apply_power()  # ←ここで即時モーター出力
        self.theta = theta
        self.pid_corrected_theta = pid_corrected_theta
        self.current_power = current_power

    def do_dist_stop(self):
        self.left_power = 0
        self.right_power = 0
        self.reset_control_values()
        self.apply_power()  # ←ここで即時モーター出力

    def do_obstacle_avoid(self):
        # --- 障害物回避動作（状態遷移あり） ---
        USER_TIME_PER_DEGREE = 1.0 / 90  # ←90度で何秒かかかるか実測値で調整
        ARC_POWER = 50
        ARC_DURATION = 4.0
        TURN_ANGLE = 45
        ARC_RATIO = 0.8  # カーブ時の弱い側のパワー比
        now = time.time()
        if self.finished:
            self.left_power = 0
            self.right_power = 0
            self.reset_control_values()  # 最初に一度だけリセット
            self.apply_power()
            return
        if self.state == 0:
            # 1段階目: 左回転
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.turn_duration = TURN_ANGLE * USER_TIME_PER_DEGREE
                self.left_power = 0
                self.right_power = ARC_POWER
            else:
                if now - self.start_time >= self.turn_duration:
                    self.et.brake()
                    self.state = 1
                    self._reset_action_vars()
        elif self.state == 1:
            # 2段階目: 右弧旋回
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + ARC_DURATION
                self.left_power = ARC_POWER
                self.right_power = int(ARC_POWER * ARC_RATIO)
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 2
                    self._reset_action_vars()
        elif self.state == 2:
            # 3段階目: 左回転
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.turn_duration = TURN_ANGLE * USER_TIME_PER_DEGREE
                self.left_power = 0
                self.right_power = ARC_POWER
            else:
                if now - self.start_time >= self.turn_duration:
                    self.et.brake()
                    self.state = 3
                    self.finished = True
                    self._reset_action_vars()
        elif self.state == 3:
            # state3: 回避完了後、即座にLINE_TRACEへ戻す
            self.finished = True
            # ここで何もしない（ModeManager側でLINE_TRACEへ遷移）
        self.reset_control_values()  # 最初に一度だけリセット
        self.apply_power()  # ←ここで即時モーター出力

    def test_initial_sensor(self, test_count=5, delay=0.2):
        """
        SPIKEの初期センサーテストを簡易実行
        """
        print("初期センサーテスト...")
        for i in range(test_count):
            test_status = self.et.get_spike_status()
            if test_status and test_status.sensors:
                color = test_status.sensors.color
                dist = test_status.sensors.distance
                color_str = f"R:{color.reflected} A:{color.ambient} C:{color.color}" if color else "N/A"
                dist_str = f"{dist}cm" if dist is not None else "N/A"
                print(f"{i+1}: OK  Color={color_str}  US={dist_str}")
            else:
                print(f"{i+1}: spike_status取得失敗")
            time.sleep(delay)
        print("初期センサーテスト完了")

    def test_arm(self):
        """
        アームを1秒上げて1秒下げる動作テスト
        """
        try:
            print("Arm up...")
            self.et.move_arm(1)  # 1 = up
            time.sleep(1.0)
            print("Arm down...")
            self.et.move_arm(0)  # 0 = down
            time.sleep(1.0)
        except Exception as e:
            print(f"Arm move error: {e}")

    def update_sensor_info(self, sensor_recorder=None):
        """
        Spikeの最新センサーステータス・カラー・超音波・モーター情報をまとめて取得し、インスタンス変数に格納
        """
        spike_status = self.et.get_spike_status()
        sensors = spike_status.sensors
        self.color = sensors.color if sensors else None
        self.distance = sensors.distance if sensors else None
        self.left_relative_position = (
            spike_status.motors['B'].relative_position
            if 'B' in spike_status.motors and spike_status.motors['B'].relative_position is not None else 0
        )
        self.right_relative_position = (
            spike_status.motors['A'].relative_position
            if 'A' in spike_status.motors and spike_status.motors['A'].relative_position is not None else 0
        )
        if sensor_recorder is not None:
            try:
                sensor_recorder.log_frame_data(spike_status)
            except Exception as e:
                import traceback
                print(f"[SensorRecorder] log_frame_data error: {e}")
                traceback.print_exc()
        # returnは不要

    def brake_for_duration(self, duration=3.0):
        """
        Spike本体に一定時間ブレーキ信号を連続送信し、安全停止を強制する
        """
        print(f"[SAFETY] Sending brake command to Spike for {duration} seconds (brake_for_duration)")
        stop_start_time = time.time()
        while time.time() - stop_start_time < duration:
            try:
                self.et.brake()
                time.sleep(0.1)
            except Exception as e:
                print(f"[SAFETY][ERROR] Exception during brake command: {e}")
                break
        print("[SAFETY] Brake command transmission completed (brake_for_duration)")

# --- システム初期化 ---
def initialize_system(record_sensor_data, save_camera_video):
    """
    ロボット・PID・センサーレコーダ・ビデオ・ソケット等の初期化をまとめて行う
    """
    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = (
        time.strftime("%Y%m%d%H%M%S", time.localtime())
        if (record_sensor_data or save_camera_video)
        else None
    )

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()

    # Initialize video writer conditionally
    video_writer = None
    video_filename = None
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(IMAGE_WIDTH, IMAGE_HEIGHT),
        )
    # Socket connection for sending camera capture
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP_ADDRESS, 8485))

    calc = ControlCalculator(
        # steer_by_camera(frame):
        #   入力画像からラインの重心座標(mx, my)、オフセットピクセル、最大輪郭を検出
        ROI_OPENCV,
        IMAGE_WIDTH,
        # calculate_theta_from_pixels(offset_pixels):
        #   ピクセル→theta変換感度
        SENSITIVITY,
        # calculate_adaptive_speed(abs_theta): theta角度（進行方向の絶対値, ラジアン）に応じて速度（パワー）を自動調整
        #     （直線・緩カーブ・急カーブで推奨パワーを自動切替, センサー値は参照しない）
        BASE_POWER,
        CURVE_POWER,
        STRAIGHT_POWER,
        # カーブ判定閾値（CURVE_THRESHOLD_DEG, ラジアンに変換）
        math.radians(CURVE_THRESHOLD_DEG),
        # 直線判定閾値（STRAIGHT_THRESHOLD_DEG, ラジアンに変換）
        math.radians(STRAIGHT_THRESHOLD_DEG)  # 直線判定のしきい値
    )

    print("メインループで直接センサー値を取得します")

    return calc, sensor_recorder, video_writer, video_filename, client_socket


# --- 可視化フレーム生成（輪郭描画含む） ---
def create_visualization_frame(frame, info, roi, mx, my, max_contour):
    """
    可視化フレームを生成し、輪郭があれば描画する
    roi: (x1, y1, x2, y2) タプル
    """
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    gray = draw_driving_info(gray, info, roi)
    if max_contour is not None:
        # Adjust contour coordinates to full frame
        adjusted_contour = max_contour + np.array([x1, y1])
        cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)  # Draw centroid
        cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)
    return gray

# --- カメラ画像送信 ---
def send_camera_capture(gray, client_socket):
    """
    カメラ画像をリモート監視用に送信
    """
    try:
        #ret, buffer = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        ret, buffer = cv2.imencode(".png", gray)
        img_encoded = buffer.tobytes()
        data = pickle.dumps(img_encoded)
        client_socket.sendall(struct.pack("L", len(data)) + data)
    except Exception as e:
        print(f"Socket error: {e}")
        return False
    return True

def prepare_driving_info(roi, mx, my, offset_pixels, mode, action, max_contour):
    """
    可視化用の走行情報を生成
    roi: (x1, y1, x2, y2) タプル
    mode: ModeManagerインスタンス
    action: ActionManagerインスタンス
    """
    x1, y1, x2, y2 = roi
    def to_distance_cm(pos):
        return int(float(pos) * 0.0471) if pos is not None else 0
    left_distance_cm = to_distance_cm(action.left_relative_position)
    right_distance_cm = to_distance_cm(action.right_relative_position)
    color = action.color
    distance = action.distance
    if color:
        color_reflected = color.reflected
        color_ambient = color.ambient
        color_color = color.color
        color_data = f"R:{color_reflected if color_reflected is not None else 'N/A'} A:{color_ambient if color_ambient is not None else 'N/A'} C:{color_color if color_color is not None else 'N/A'}"
    else:
        color_data = "R:N/A A:N/A C:N/A"
    if distance is not None:
        ultrasonic_data = f"{distance} cm"
    else:
        ultrasonic_data = "N/A cm"
    info = dict()
    info["offset_x"], info["offset_y"] = x1 + mx, y1 + my
    info["roi"] = roi
    info["text"] = {
        "offset_pixels": f"{round(offset_pixels, 1)}px",
        "theta_deg": f"{round(math.degrees(action.theta), 2)}deg",
        "pid_corrected_theta": f"{round(math.degrees(action.pid_corrected_theta), 2)}deg",
        "power_status": (
            "OFF_LINE" if action.theta == 0 else
            "CURVE" if abs(action.theta) > math.radians(CURVE_THRESHOLD_DEG) else
            "STRAIGHT" if abs(action.theta) < math.radians(STRAIGHT_THRESHOLD_DEG) else
            "BASE"
        ),
        "current_power": f"{round(action.current_power, 1)}%",
        "on_color": (
            "BLACK" if (color and color.is_black) else ("BLUE" if (color and color.is_blue) else "N/A")
        ),
        "color_sensor": color_data,
        "ultrasonic_sensor": ultrasonic_data,
        "left_power": f"{action.left_power}%",
        "right_power": f"{action.right_power}%",
        "left_relative_position": f"{action.left_relative_position}deg / {left_distance_cm}cm",
        "right_relative_position": f"{action.right_relative_position}deg / {right_distance_cm}cm",
        "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
        "mode": mode.mode.name if hasattr(mode, 'mode') else str(mode)
    }
    return info

# --- メイン処理 ---
def main(record_sensor_data=False, save_camera_video=False):
    """
    メイン制御ループ
    - 各種初期化（ロボット・センサーレコーダ・ビデオ・ソケット等）
    - 初期センサーテスト・アーム動作テスト
    - メインループで以下を繰り返す：
        1. カメラ画像取得
        2. ActionManager経由でセンサー情報取得（カラー・超音波・モーター相対位置/パワー）
        3. （必要に応じて）AIモデル推論例（画像・distance・motor_infoを入力、center_xとobject_detectedを出力）
        4. 画像処理でライン重心・オフセット・最大輪郭を検出
        5. モード遷移・制御出力（ModeManager/ActionManager）
           - LINE_TRACE: ライントレース制御
           - DIST_STOP: 障害物検知時の一時停止
           - OBSTACLE_AVOID: 障害物回避動作
           - 必要に応じてGOALやSMART_CARRY等も拡張可
        6. モードごとの制御値をActionManagerから取得し、可視化情報を生成
        7. 可視化フレーム生成（輪郭・重心描画など）
        8. カメラ画像の送信・保存（リモート監視や動画保存）
        9. ループ周期調整（30ms未満ならsleepで調整）
        10. 例外・割り込み時は安全停止（brake/stop）・リソース解放

    - 各処理はActionManager/ModeManager/ControlCalculator等の責務に分離し、
      mainは「全体の流れ・状態遷移・例外処理・リソース管理」のみを記述
    - AIモデル推論例は「画像・distance・motor_info→center_x, object_detected」設計例をコメントで明記
    - 例外発生時も必ず安全停止・リソース解放を徹底
    """
    calc, sensor_recorder, video_writer, video_filename, client_socket = initialize_system(
        record_sensor_data, save_camera_video
    )
    time.sleep(0.5)

    mode = ModeManager()
    action = ActionManager()  # etはActionManager内で生成

    # --- 初期動作テスト ---
    action.test_initial_sensor()  # SPIKEの初期センサーテスト
    action.test_arm()            # アーム動作テスト

    # --- mainループ ---
    try:
        while action.et.is_running == True:
            loop_start = time.time()
            # 1. カメラ画像取得
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Can't receive frame (stream end?). Exiting ...")
                break
            # 2. センサー情報取得（ActionManager経由で一括取得）
            action.update_sensor_info(sensor_recorder)
            # 3. （必要に応じて）AIモデル推論例（コメント参照）
            # 例: ニューラルネットワークによる進行方向推定や物体検出を組み込む場合は、
            # 必要な入力（画像、distance、motor_infoなど）をaction.update_by_nn等で渡し、
            # 推論結果として「進行方向x座標（center_x）」および「物体検出結果（object_detected）」の2つを返す設計が推奨されます。
            #
            # 例（コメントアウト）：
            #   nn_result = action.update_by_nn(
            #       frame=frame,
            #       distance=action.distance,
            #       motor_info=action.motor_info
            #   )
            #   # nn_resultの内容（例: {'center_x': ..., 'object_detected': ...}）
            #   center_x = nn_result['center_x']
            #   object_detected = nn_result['object_detected']
            #
            # ※AIモデルの統合時は、mainループのこの位置で推論・状態更新を行うと保守性・拡張性が高まります。
            # -------------------------------------------------------------
            # 4. 画像処理でライン重心・オフセット・最大輪郭を検出
            mx, my, offset_pixels, max_contour = calc.steer_by_camera(frame)
            # 5. モード遷移・制御出力（ModeManager/ActionManager）
            mode.update_and_act(
                action,
                offset_pixels=offset_pixels,
                calc=calc
            )
            # 6. 可視化情報生成
            info = prepare_driving_info(
                ROI_OPENCV,
                mx,
                my,
                offset_pixels,
                mode,
                action,
                max_contour
            )
            # 7. 可視化フレーム生成
            gray = create_visualization_frame(
                frame,
                info,
                ROI_OPENCV,
                mx,
                my,
                max_contour
            )
            # 8. カメラ画像の送信・保存
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)
            if not send_camera_capture(gray, client_socket):
                print("[ERROR] send_camera_capture failed. Breaking main loop.")
                break
            # 9. ループ周期調整（30ms未満ならsleep）
            # elapsed = time.time() - loop_start
            # if elapsed < 0.03:
            #     time.sleep(0.03 - elapsed)
    except KeyboardInterrupt:
        # ユーザーによる割り込み（Ctrl+C）時：安全のため一定時間ブレーキ信号を連続送信
        print("Interrupted by user")
        action.brake_for_duration()  # Spikeに3秒間ブレーキ信号を送り続ける
    except Exception as e:
        # 予期しない例外発生時も必ずロボットを安全に停止（3秒間ブレーキ信号送信）し、例外内容を表示
        print(f"[ERROR] Unexpected exception: {e}")
        action.brake_for_duration()  # Spikeに3秒間ブレーキ信号を送り続ける
    finally:
        # いかなる場合もリソースを必ず解放し、安全停止を徹底
        action.et.stop()  # モーター・アクチュエータを安全停止（多重呼び出しでも安全）
        cap.release()     # カメラリソース解放
        client_socket.close()  # ソケット通信終了
        if save_camera_video and video_writer is not None:
            video_writer.release()  # 動画ファイル保存終了
            print(f"Video saved to: {video_filename}")
        if sensor_recorder is not None:
            sensor_recorder.stop_recording()  # センサーログ記録終了
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the OpenCV-based line following robot with optional sensor recording and video saving"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Record sensor data to file"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save camera video to file"
    )

    args = parser.parse_args()

    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_OPENCV}")
    print(f"Base power: {BASE_POWER}")
    print("Press Ctrl+C to stop")

    main(record_sensor_data=args.record_sensor, save_camera_video=args.save_video)
