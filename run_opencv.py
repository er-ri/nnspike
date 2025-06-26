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
"""

# ==== ユーザー調整用パラメータ（ここだけ編集すればOK） ====
# ROI_OPENCV: OpenCV画像処理で使用する領域（左上x, 左上y, 右下x, 右下y）
ROI_OPENCV = (150, 300, 490, 400)  # 必ずタプルで定義すること
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480                 # カメラ画像の高さ
BASE_POWER = 50                    # カーブ時の基準パワー
STRAIGHT_POWER = 80                # 直線時の推奨パワー
CURVE_POWER = 30                   # 急カーブ時の最低パワー
CURVE_THRESHOLD_DEG = 10           # カーブ判定閾値（度数法, SENSITIVITY=1.0時の推奨値）
STRAIGHT_THRESHOLD_DEG = 3         # 直線判定のしきい値（ユーザー調整用, デフォルト3度, STRAIGHT_THRESHOLD_DEGで指定）
SENSITIVITY = 1.0                  # ピクセル→theta変換感度
MAX_STEERING_POWER_DIFF = 40       # 最大旋回時の左右パワー差（%）
MAX_STEERING_THETA_DEG = 50        # 最大旋回角（度数法, 例: 50度）
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
    PIDController,
    SensorRecorder,
)

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


def run_initial_sensor_test(et, test_count=5, delay=0.2):
    """
    SPIKEの初期センサーテストを簡易実行
    """
    print("初期センサーテスト...")
    for i in range(test_count):
        test_status = et.get_spike_status()
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

    # Initialize robot, PID, ControlCalculator（ユーザー調整値はグローバル参照）
    et = ETRobot()
    pid = PIDController(
        Kp=2.0,  # 比例項: 反応をやや抑える（従来3→2.0）
        Ki=0,    # 積分項: 通常0でOK
        Kd=0.4,  # 微分項: 揺れ抑制を強める（従来0.2→0.4）
        setpoint=0,
        output_limits=(-0.25, 0.25),  # Direct radian limits for steering correction
    )
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

    # --- アームを1秒上げて1秒下げる（動作確認） ---
    try:
        print("Arm up...")
        et.move_arm(1)  # 1 = up
        time.sleep(1.0)
        print("Arm down...")
        et.move_arm(0)  # 0 = down
        time.sleep(1.0)
    except Exception as e:
        print(f"Arm move error: {e}")

    # 初期センサーテスト
    run_initial_sensor_test(et)

    return et, pid, calc, sensor_recorder, video_writer, video_filename, client_socket


def send_stop_signal(et, duration=3.0):
    """
    Spikeに一定時間ブレーキ信号を送り続ける
    """
    print(f"Sending stop signals to Spike for {duration} seconds...")
    stop_start_time = time.time()
    while time.time() - stop_start_time < duration:
        try:
            et.brake()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error sending stop signal: {e}")
            break
    print("Stop signal transmission completed")


def get_sensor_info(et, sensor_recorder=None):
    """
    Spikeの最新センサーステータス・カラー・超音波・モーター情報をまとめて取得
    - カラーセンサー値取得と黒・青判定
    - 超音波センサーデータ値取得
    - モーターB/C相対位置値・パワー値取得
    - センサーデータ記録が有効な場合はロガーに記録
    """
    spike_status = et.get_spike_status()
    sensors = spike_status.sensors
    color = sensors.color if sensors else None
    distance = sensors.distance if sensors else None
    motor_info = {
        'left': spike_status.motors['B'].relative_position if 'B' in spike_status.motors and spike_status.motors['B'].relative_position is not None else 0,
        'right': spike_status.motors['A'].relative_position if 'A' in spike_status.motors and spike_status.motors['A'].relative_position is not None else 0,
        'left_power': spike_status.motors['B'].power if 'B' in spike_status.motors and hasattr(spike_status.motors['B'], 'power') else 0,
        'right_power': spike_status.motors['A'].power if 'A' in spike_status.motors and hasattr(spike_status.motors['A'], 'power') else 0
    }
    if sensor_recorder is not None:
        try:
            sensor_recorder.log_frame_data(spike_status)
        except Exception as e:
            import traceback
            print(f"[SensorRecorder] log_frame_data error: {e}")
            traceback.print_exc()
    return color, distance, motor_info


def prepare_driving_info(roi, mx, my, offset_pixels, theta, pid_corrected_theta, current_power, left_power, right_power, color, distance, relative_position, max_contour, mode=None):
    """
    可視化用の走行情報を生成
    roi: (x1, y1, x2, y2) タプル
    mode: 現在の動作モード（例: 'LINE_TRACE'）
    """
    x1, y1, x2, y2 = roi
    def to_distance_cm(pos):
        return int(float(pos) * 0.0471) if pos is not None else 0
    left_distance_cm = to_distance_cm(relative_position['left'])
    right_distance_cm = to_distance_cm(relative_position['right'])
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
        "theta_deg": f"{round(math.degrees(theta), 2)}deg",
        "pid_corrected_theta": f"{round(math.degrees(pid_corrected_theta), 2)}deg",
        "power_status": (
            "OFF_LINE" if theta == 0 else
            "CURVE" if abs(theta) > math.radians(CURVE_THRESHOLD_DEG) else
            "STRAIGHT" if abs(theta) < math.radians(STRAIGHT_THRESHOLD_DEG) else
            "BASE"
        ),
        "current_power": f"{round(current_power, 1)}%",
        "on_color": (
            "BLACK" if (color and color.is_black) else ("BLUE" if (color and color.is_blue) else "N/A")
        ),
        "color_sensor": color_data,
        "ultrasonic_sensor": ultrasonic_data,
        "left_power": f"{left_power}%",
        "right_power": f"{right_power}%",
        "left_relative_position": f"{relative_position['left']}deg / {left_distance_cm}cm",
        "right_relative_position": f"{relative_position['right']}deg / {right_distance_cm}cm",
        "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
        "mode": mode if mode is not None else "N/A"
    }
    return info

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
        ret, buffer = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_encoded = buffer.tobytes()
        data = pickle.dumps(img_encoded)
        client_socket.sendall(struct.pack("L", len(data)) + data)
    except Exception as e:
        print(f"Socket error: {e}")
        return False
    return True

# --- 動作モード管理クラス ---
from enum import Enum, auto

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
    OBSTACLE_DETECT_DISTANCE = 50  # 障害物検知のしきい値[cm]
    DIST_STOP_DURATION = 2.0       # 距離停止モードの待機時間[秒]
    def __init__(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None
    def update(self, distance):
        # 距離センサー値に応じてモード遷移
        if self.mode == Mode.LINE_TRACE:
            if distance is not None and distance < self.OBSTACLE_DETECT_DISTANCE:
                self.mode = Mode.DIST_STOP
                self.obstacle_detected_time = time.time()
        elif self.mode == Mode.DIST_STOP:
            if distance is not None and distance >= self.OBSTACLE_DETECT_DISTANCE * (50/30):
                self.mode = Mode.LINE_TRACE
                self.obstacle_detected_time = None
            elif time.time() - self.obstacle_detected_time >= self.DIST_STOP_DURATION:
                self.mode = Mode.OBSTACLE_AVOID
        elif self.mode == Mode.OBSTACLE_AVOID:
            pass
        # SMART_CARRY_1, SMART_CARRY_2, GOALへの遷移は必要に応じて追加
    def reset(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None

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
    USER_TIME_PER_DEGREE = 1.0 / 90  # ←90度で何秒かかかるか実測値で調整
    ARC_POWER = 50
    ARC_DURATION = 5.0
    TURN_ANGLE = 45
    ARC_RATIO = 0.8  # カーブ時の弱い側のパワー比
    def __init__(self, et):
        self.et = et
        self.state = 0
        self.start_time = None
        self.finished = False
        self.action_sent = False
    def reset(self):
        self.state = 0
        self.start_time = None
        self.finished = False
        self.action_sent = False
    def obstacle_avoid_step(self):
        # 障害物回避動作の非ブロッキング実装: sleep等のブロック処理は絶対に入れない
        now = time.time()
        if self.finished:
            return
        if self.state == 0:
            if not self.action_sent:
                self.et.turn_left(degree=self.TURN_ANGLE, power=self.ARC_POWER, time_per_degree=self.USER_TIME_PER_DEGREE)
                self.start_time = now
                self.action_sent = True
            else:
                duration = self.TURN_ANGLE * self.USER_TIME_PER_DEGREE
                if now - self.start_time >= duration:
                    self.et.brake()
                    self.state = 1
                    self.action_sent = False
        elif self.state == 1:
            if not self.action_sent:
                self.et.move_right_arc(duration=self.ARC_DURATION, power=self.ARC_POWER, ratio=self.ARC_RATIO)
                self.start_time = now
                self.action_sent = True
            else:
                if now - self.start_time >= self.ARC_DURATION:
                    self.et.brake()
                    self.state = 2
                    self.action_sent = False
        elif self.state == 2:
            if not self.action_sent:
                self.et.turn_left(degree=self.TURN_ANGLE, power=self.ARC_POWER, time_per_degree=self.USER_TIME_PER_DEGREE)
                self.start_time = now
                self.action_sent = True
            else:
                duration = self.TURN_ANGLE * self.USER_TIME_PER_DEGREE
                if now - self.start_time >= duration:
                    self.et.brake()
                    self.state = 3
                    self.action_sent = False
        elif self.state == 3:
            self.finished = True
    def is_finished(self):
        return self.finished


# --- メイン処理 ---
def main(record_sensor_data=False, save_camera_video=False):
    et, pid, calc, sensor_recorder, video_writer, video_filename, client_socket = initialize_system(
        record_sensor_data, save_camera_video
    )
    time.sleep(0.5)
    et.set_motor_relative_position(left_position=0, right_position=0)

    # モード管理クラス・固有動作管理クラスのインスタンス生成
    mode_manager = ModeManager()
    action_manager = ActionManager(et)

    # --- 直前の画像処理結果を保持する変数を初期化 ---
    mx = my = offset_pixels = theta = pid_corrected_theta = current_power = left_power = right_power = 0
    max_contour = None

    try:
        while et.is_running == True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Can't receive frame (stream end?). Exiting ...")
                break
            # Spikeの最新センサーステータス・カラー・超音波・モーター相対位置値を取得
            color, distance, motor_info = get_sensor_info(et, sensor_recorder)
            # --- モデル入出力設計例 ---
            # ▼ニューラルネット統合例（必要に応じて有効化）
            # 入力: 画像, motor_info（相対位置などを含むdict）, 超音波センサー値
            # 出力: direction_coords（進行方向のx座標）, object_detected（0=なし, 1=ペットボトル, 2=交差点）
            # 例:
            # roi_area = process_image(
            #     image=frame.copy(),
            #     device=device,         # 推論デバイス（例: 'cpu' or 'cuda'）
            #     roi=ROI_CNN            # モデル用ROI（必要に応じて指定）
            # )
            # # Todo: ステージ判定
            # interval_idx = 0
            # with torch.no_grad():  # ニューラルネット推論時のみ必要
            #     direction_coords, object_detected = models[interval_idx](
            #         roi_area,
            #         motor_info,  # 相対位置情報などを含むdict
            #         distance if distance is not None else 0
            #     )
            #     # direction_coords: 進行方向のx座標（単一値）
            #     # object_detected: 前方物体判定（0=なし, 1=ペットボトル, 2=交差点）
            #     mode_manager.update(motor_info, distance, object_detected)
            #
            # ※torch.no_grad()はニューラルネット推論時のみ必要。OpenCVのみの場合は不要。
            mx, my, offset_pixels, max_contour = calc.steer_by_camera(frame)
            # --- MEMO: object_detectedは将来的にモデル出力や画像処理で取得し、mode_manager.update()に渡す ---
            object_detected = None  # 例: 0=なし, 1=ペットボトル, 2=交差点/ゴール
            mode_manager.update(distance, object_detected)
            # --- モードごとの処理 ---
            if mode_manager.mode == Mode.LINE_TRACE:
                # 進行角度thetaを計算
                theta = calc.calculate_theta_from_pixels(offset_pixels)
                # 推奨速度を計算（カーブ時は減速）
                current_power = calc.calculate_adaptive_speed(abs(theta))
                # PID制御でthetaを補正し、左右パワー差を計算
                pid_corrected_theta = pid.update(theta)
                max_theta = math.radians(MAX_STEERING_THETA_DEG)
                power_adjustment = int((pid_corrected_theta / max_theta) * MAX_STEERING_POWER_DIFF)
                left_power = int(current_power - power_adjustment)
                right_power = int(current_power + power_adjustment)
                et.set_motor_forward_power(left_power=left_power, right_power=right_power)
            elif mode_manager.mode == Mode.DIST_STOP:
                # 一時停止し、2秒間DIST_STOPモードで距離の再確認のみ行う（前進しない）
                et.brake()
                time.sleep(0.01)  # しっかり停止しCPU負荷も下げる
                theta = pid_corrected_theta = current_power = 0
                left_power = right_power = 0
            elif mode_manager.mode == Mode.OBSTACLE_AVOID:
                # 障害物回避動作（オブスタクルボトル回避）のみ実行、実際のモーター出力値をspike_statusから反映
                # ※ action_manager.step()は必ず非ブロッキングで設計すること（sleep等を入れない）
                theta = pid_corrected_theta = current_power = 0
                action_manager.obstacle_avoid_step()
                left_power = motor_info['left_power'] if 'left_power' in motor_info else 0
                right_power = motor_info['right_power'] if 'right_power' in motor_info else 0
                if action_manager.is_finished():
                    mode_manager.reset()
            elif mode_manager.mode == Mode.SMART_CARRY_1:
                # スマートキャリー1回目の処理（必要に応じて実装）
                pass
            elif mode_manager.mode == Mode.SMART_CARRY_2:
                # スマートキャリー2回目の処理（必要に応じて実装）
                pass
            elif mode_manager.mode == Mode.GOAL:
                # ゴール到達時の固有動作（例: 完全停止・アーム動作等）
                action_manager.step(mode=Mode.GOAL)
                left_power = right_power = 0
                theta = pid_corrected_theta = current_power = 0
                # MEMO: 完了判定後に必要ならmode_manager.reset()等で再スタート可能
            # --- ここから共通処理 ---
            # 可視化用情報生成
            info = prepare_driving_info(ROI_OPENCV, mx, my, offset_pixels, theta, pid_corrected_theta, current_power, left_power, right_power, color, distance, motor_info, max_contour, mode=mode_manager.mode.name)
            # 可視化フレーム生成
            gray = create_visualization_frame(frame, info, ROI_OPENCV, mx, my, max_contour)
            # --- ここで必ずカメラ画像を送信・保存 ---
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)
            if not send_camera_capture(gray, client_socket):
                print("[ERROR] send_camera_capture failed. Breaking main loop.")
                break

            # --- ループ周期制御: 1サイクル30ms未満ならsleepで調整 ---
            elapsed = time.time() - loop_start
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)
    except KeyboardInterrupt:
        print("Interrupted by user")
        send_stop_signal(et, duration=3.0)
    except Exception as e:
        # print(f"[EXCEPTION] Error: {e}")
        # traceback.print_exc()
        send_stop_signal(et, duration=3.0)
    finally:
        et.stop()
        cap.release()
        client_socket.close()
        if save_camera_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")
        if sensor_recorder is not None:
            sensor_recorder.stop_recording()
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
