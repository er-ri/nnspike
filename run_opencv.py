#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

This script controls a line-following robot using OpenCV computer vision
instead of neural network predictions. It uses the steer_by_camera function
to detect the line centroid and follows it using PID control.

Speed Tuning Parameters:
- BASE_POWER: Base power for straight lines (start here)
- TURN_REDUCTION_FACTOR: Speed reduction in turns (0.0-1.0)
- TURN_THRESHOLD: Pixel offset threshold to detect turns

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
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

# Socket connection settings
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


# AsyncSensorReader クラスを削除 - メインループで直接センサー値を取得するように変更


def run_initial_sensor_test(et, test_count=5, delay=0.2):
    """SPIKEの初期センサーテストを簡素に実行"""
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
    """ロボット・PID・センサーレコーダ・ビデオ・ソケット等の初期化をまとめて行う"""
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

    # --- アームを1秒上げて1秒下げる処理を追加（test_color_only.py参考） ---
    try:
        print("Arm up...")
        et.move_arm(1)  # 1 = up
        time.sleep(1.0)
        print("Arm down...")
        et.move_arm(0)  # 0 = down
        time.sleep(1.0)
    except Exception as e:
        print(f"Arm move error: {e}")

    # 初期センサーテストを関数で実行
    run_initial_sensor_test(et)

    return et, pid, calc, sensor_recorder, video_writer, video_filename, client_socket


def send_stop_signal(et, duration=3.0):
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
    Spikeの最新センサーステータス・カラー・超音波・モーター相対位置値・判定をまとめて取得
    - Spikeの最新センサーステータスを取得
    - カラーセンサー値取得と黒・青判定
    - 超音波センサーデータ値取得
    - モーターB/C相対位置値取得
    - センサーデータ記録が有効な場合はロガーに記録
    """
    spike_status = et.get_spike_status()
    sensors = spike_status.sensors
    color = sensors.color if sensors else None
    distance = sensors.distance if sensors else None
    relative_position = {
        'left': spike_status.motors['B'].relative_position if 'B' in spike_status.motors and spike_status.motors['B'].relative_position is not None else 0,
        'right': spike_status.motors['A'].relative_position if 'A' in spike_status.motors and spike_status.motors['A'].relative_position is not None else 0
    }
    if sensor_recorder is not None:
        sensor_recorder.log_frame_data(spike_status)
    return color, distance, relative_position


def prepare_driving_info(roi, mx, my, offset_pixels, theta, pid_corrected_theta, current_power, left_power, right_power, color, distance, relative_position, max_contour):
    """
    可視化用の走行情報を生成
    roi: (x1, y1, x2, y2) タプル
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
    }
    return info

# --- 可視化フレーム生成（輪郭描画含む） ---
def create_visualization_frame(frame, info, roi, mx, my, max_contour):
    """
    Create visualization frame and draw contour on the visualization if found
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
    Send camera capture for remote monitoring
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
    LINE_TRACE = auto()
    SEMI_AVOID = auto()
    OBSTACLE_AVOID = auto()

class ModeManager:
    """
    動作モードの状態遷移を管理するクラス。
    LINE_TRACE: 通常のライン走行
    SEMI_AVOID: 障害物検知後の一時停止・確認（低速前進し距離変化を監視）
    OBSTACLE_AVOID: 障害物回避動作
    """
    OBSTACLE_DETECT_DISTANCE = 40  # 障害物検知のしきい値（cm, デフォルト30）
    SEMI_AVOID_DURATION = 2.0      # セミ回避モードの待機秒数（デフォルト2.0秒）
    def __init__(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None
        self.avoid_start_time = None
        self.last_distance = None
    def update(self, distance):
        if self.mode == Mode.LINE_TRACE:
            if distance is not None and distance < self.OBSTACLE_DETECT_DISTANCE:
                self.mode = Mode.SEMI_AVOID
                self.obstacle_detected_time = time.time()
                self.last_distance = distance
        elif self.mode == Mode.SEMI_AVOID:
            # 2秒経過したらOBSTACLE_AVOIDへ（距離が縮まっても即移行しない）
            if distance is not None and distance >= self.OBSTACLE_DETECT_DISTANCE * (50/30):
                self.mode = Mode.LINE_TRACE
                self.obstacle_detected_time = None
                self.last_distance = None
            elif time.time() - self.obstacle_detected_time >= self.SEMI_AVOID_DURATION:
                self.mode = Mode.OBSTACLE_AVOID
                self.avoid_start_time = time.time()
            # 距離の更新は維持（次回の判定用）
            self.last_distance = distance if distance is not None else self.last_distance
        elif self.mode == Mode.OBSTACLE_AVOID:
            pass
    def reset(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None
        self.avoid_start_time = None
        self.last_distance = None

# --- 固有動作管理クラス（回避・今後の特殊動作用） ---
class ActionManager:
    """
    障害物回避などの固有動作を管理する拡張用クラス。
    例: 45度左回転→右弧旋回→45度左回転で回避（カラーセンサーは使わず、距離・時間・カーブ比のみで制御）
    """
    USER_TIME_PER_DEGREE = 1.0 / 90  # ←90度で何秒かかるか実測値で調整
    ARC_POWER = 50
    ARC_DURATION = 5.0
    TURN_ANGLE = 45
    ARC_RATIO = 0.8  # カーブ時の弱い側のパワー比
    def __init__(self, et):
        self.et = et
        self.state = 0
        self.start_time = None
        self.finished = False
    def reset(self):
        self.state = 0
        self.start_time = None
        self.finished = False
    def step(self):
        # 回避動作：
        # 1. 45度左回転
        # 2. 右弧旋回
        # 3. 45度左回転
        # 4. ライントレースモードに切り替え
        if self.finished:
            return
        if self.state == 0:
            self.et.turn_left(degree=self.TURN_ANGLE, power=self.ARC_POWER, time_per_degree=self.USER_TIME_PER_DEGREE)
            self.state = 1
        elif self.state == 1:
            self.et.move_right_arc(duration=self.ARC_DURATION, power=self.ARC_POWER, ratio=self.ARC_RATIO)
            self.state = 2
        elif self.state == 2:
            self.et.turn_left(degree=self.TURN_ANGLE, power=self.ARC_POWER, time_per_degree=self.USER_TIME_PER_DEGREE)
            self.state = 3
        elif self.state == 3:
            self.finished = True  # ここで回避動作終了
    def is_finished(self):
        return self.finished


def main(record_sensor_data=False, save_camera_video=False):
    et, pid, calc, sensor_recorder, video_writer, video_filename, client_socket = initialize_system(
        record_sensor_data, save_camera_video
    )
    time.sleep(0.5)
    et.set_motor_relative_position(left_position=0, right_position=0)

    # モード管理クラス・固有動作管理クラスのインスタンス生成
    mode_manager = ModeManager()
    action_manager = ActionManager(et)

    try:
        while et.is_running == True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # ビデオ保存有効時はフレームを保存
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)
            # Spikeの最新センサーステータス・カラー・超音波・モーター相対位置値を取得
            color, distance, relative_position = get_sensor_info(et, sensor_recorder)
            # モード更新
            mode_manager.update(distance)
            # --- 各モードでの走行パラメータ初期化 ---
            mx = my = offset_pixels = theta = pid_corrected_theta = current_power = left_power = right_power = 0
            max_contour = None
            # --- モードごとの処理 ---
            if mode_manager.mode == Mode.LINE_TRACE:
                # ラインの重心座標・オフセット・最大輪郭を取得
                mx, my, offset_pixels, max_contour = calc.steer_by_camera(frame)
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
            elif mode_manager.mode == Mode.SEMI_AVOID:
                # 一時停止し、2秒間セミ回避モードで距離の再確認のみ行う（前進しない）
                left_power = right_power = 0
            elif mode_manager.mode == Mode.OBSTACLE_AVOID:
                # 固有動作ステップ実行（カラーセンサーは使わない）
                action_manager.step()
                left_power = right_power = 0
                if action_manager.is_finished():
                    mode_manager.reset()
            # --- ここから共通処理 ---
            # 計算したパワーでモーターを駆動
            et.set_motor_forward_power(left_power=left_power, right_power=right_power)
            # 可視化用情報生成
            info = prepare_driving_info(ROI_OPENCV, mx, my, offset_pixels, theta, pid_corrected_theta, current_power, left_power, right_power, color, distance, relative_position, max_contour)
            # 可視化フレーム生成
            gray = create_visualization_frame(frame, info, ROI_OPENCV, mx, my, max_contour)
            # カメラ画像送信（リモートモニタ用）
            if not send_camera_capture(gray, client_socket):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
        send_stop_signal(et, duration=5.0)
    except Exception as e:
        print(f"Error: {e}")
        send_stop_signal(et, duration=5.0)
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
