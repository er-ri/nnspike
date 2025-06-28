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
BASE_POWER = 80                    # カーブ時の基準パワー 40→45にUP
STRAIGHT_POWER = 120                # 直線時の推奨パワー 50→60にUP
CURVE_POWER = 50                   # 急カーブ時の最低パワー 30→35にUP
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
import traceback
from nnspike.unit import ETRobot
from nnspike.utils.control import ControlCalculator
from nnspike.utils import (
    draw_driving_info,
    SensorRecorder,
    PIDController,
)
from enum import Enum, auto
from dataclasses import dataclass
import sys
import time
try:
    import msvcrt
    WINDOWS = True
except ImportError:
    import termios
    import tty
    import select
    WINDOWS = False

@dataclass
class Config:
    log_sensor: bool = False
    log_save_video: bool = False
    log_send_video: bool = False
    log_manual: bool = False
    # 必要に応じて他の設定も追加

# --- ソケット通信設定 ---
HOST_IP_ADDRESS = (
    "192.168.137.1"  # Raspberry PiがPCへ送信する宛先IP
)

class Mode(Enum):
    LINE_TRACE = auto()         # 通常のライン走行
    DIST_STOP = auto()          # 超音波距離停止モード（障害物検知後の一時停止・距離再確認）
    OBSTACLE_AVOID = auto()     # 障害物回避動作（オブスタクルボトル回避）
    SMART_CARRY_1 = auto()      # スマートキャリー1回目
    SMART_CARRY_2 = auto()      # スマートキャリー2回目
    GOAL = auto()               # ゴール到達モード（ゴールに向かう処理とゴール停止をこのモードで実装する構想）
    MANUAL = auto()             # マニュアル操作モード（手動制御用）
    STOP = auto()             # マニュアル操作モード（手動制御用）
    MANUAL_A = auto()             # マニュアル操作モード（手動制御用）
    MANUAL_B = auto()             # マニュアル操作モード（手動制御用）
    MANUAL_C = auto()             # マニュアル操作モード（手動制御用）

class ActionManager:
    """
    障害物回避やゴール到達時などの固有動作を管理する拡張用クラス。
    例: 45度左回転→右弧旋回→45度左回転で回避（非ブロッキングで各動作を進める）
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
        self.calc = ControlCalculator(
            BASE_POWER,
            CURVE_POWER,
            STRAIGHT_POWER,
            math.radians(CURVE_THRESHOLD_DEG),
            math.radians(STRAIGHT_THRESHOLD_DEG)
        )
        self.et.set_motor_relative_position(left_position=0, right_position=0)
        self.reset_control_values()  # theta, pid_corrected_theta, current_powerをまとめてリセット
        # --- 受信した実際の左右パワー値を初期化 ---
        self.left_actual_power = None
        self.right_actual_power = None

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
        """現在のleft_power, right_powerをロボットに反映し、spikeから実際の左右パワーを取得"""
        self.et.set_motor_forward_power(left_power=self.left_power, right_power=self.right_power)
        # spikeから実際の左右パワーを取得
        status = self.et.get_spike_status()
        left_actual = None
        right_actual = None
        if status and hasattr(status, 'motors'):
            # B:左, A:右（設計に応じて要確認）
            left_actual = status.motors['B'].power if 'B' in status.motors and hasattr(status.motors['B'], 'power') else None
            right_actual = status.motors['A'].power if 'A' in status.motors and hasattr(status.motors['A'], 'power') else None
        # --- 取得した値をインスタンス変数に格納 ---
        self.left_actual_power = left_actual
        self.right_actual_power = right_actual

    def reset_control_values(self):
        self.theta = 0
        self.pid_corrected_theta = 0
        self.current_power = 0

    def do_line_trace(self, offset_pixels):
        # --- ライントレース時の進行角度・推奨速度・PID補正・左右パワー計算 ---
        # MAX_THETA_DEG: 最大旋回角（度数法, ユーザー調整パラメータで一元管理）
        # MAX_POWER_DIFF: 最大旋回時の左右パワー差（%）, ユーザー調整パラメータで一元管理
        # offset_pixels: ライン重心のオフセット（ピクセル単位, 画像中心からのズレ）
        theta = self.calc.calculate_theta_from_pixels(offset_pixels, IMAGE_WIDTH, SENSITIVITY)  # オフセットピクセル→進行角度（ラジアン）へ変換
        current_power = self.calc.calculate_adaptive_speed(abs(theta))  # 進行角度に応じて推奨速度（パワー）を自動調整
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

    def do_stop(self):
        self.left_power = 0
        self.right_power = 0
        self.apply_power()

    def do_straight(self):
        """
        直進するだけのアクション
        """
        self.left_power = BASE_POWER
        self.right_power = BASE_POWER
        self.apply_power()

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
            self.reset_control_values()  # 毎回リセット（安全のため）
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
        elif self.state == 3:
            pass
        self.reset_control_values()
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
        if sensor_recorder is not None and sensor_recorder.is_enabled():
            try:
                sensor_recorder.log(spike_status)
            except Exception as e:
                print(f"[SensorRecorderManager] log error: {e}")
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

class Camera:
    """
    カメラ操作をカプセル化するクラス。
    cap.read() などのOpenCVカメラ操作を分離し、mainから直接触らない設計。
    画像取得と画像処理（steer_by_camera）も一元化。
    """
    def __init__(self, device_index=0, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=30, roi=ROI_OPENCV):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.roi = roi
        self.image_width = width

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def steer_by_camera(self, frame):
        """
        カメラフレームからROI内の輪郭検出を行い、進行方向の判断に必要な情報を辞書で返す。
        """
        x1, y1, x2, y2 = self.roi
        roi_area = frame[y1:y2, x1:x2]
        image = cv2.cvtColor(roi_area, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            mu = cv2.moments(max_contour)
            mx = mu["m10"] / (mu["m00"] + 1e-5)
            my = mu["m01"] / (mu["m00"] + 1e-5)
        else:
            mx = image.shape[1] / 2
            my = image.shape[0] / 2
            max_contour = None
        roi_center_x = image.shape[1] / 2
        offset_pixels = mx - roi_center_x
        return {
            "mx": mx,
            "my": my,
            "offset_pixels": offset_pixels,
            "max_contour": max_contour
        }

class SensorRecorderManager:
    """
    SensorRecorderの生成・管理・利用を一元化するクラス。
    mainやActionManager等からはこのクラス経由でセンサーログ記録を行う。
    """
    def __init__(self, enable_recording: bool):
        self.enable_recording = enable_recording
        self.recorder = None
        if enable_recording:
            timestamp = get_timestamp()
            self.recorder = SensorRecorder(timestamp=timestamp)
            self.recorder.start_recording()

    def log(self, spike_status):
        if self.recorder is not None:
            self.recorder.log_frame_data(spike_status)

    def stop(self):
        if self.recorder is not None:
            self.recorder.stop_recording()

    def get_frame_count(self):
        if self.recorder is not None:
            return self.recorder.get_frame_count()
        return 0

    def is_enabled(self):
        return self.enable_recording

class VideoManager:
    """
    カメラ動画保存とソケット通信の初期化・管理を一元化するクラス。
    可視化フレーム生成や走行情報生成も担当する。
    """
    def __init__(self, save_video, send_video, image_width, image_height, host_ip_address, port=8485):
        self.video_writer = None
        self.video_filename = None
        self.client_socket = None
        self.save_video = save_video
        self.send_video = send_video
        if save_video:
            timestamp = get_timestamp()
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_filename = f"storage/videos/{timestamp}_picamera.avi"
            self.video_writer = cv2.VideoWriter(
                filename=self.video_filename,
                fourcc=fourcc,
                fps=30,
                frameSize=(image_width, image_height),
            )
        if send_video:
            # --- 注意: PC側でレシーバ（サーバ）が起動していない場合、ここで例外が発生しプログラムは停止します ---
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((host_ip_address, port))

    def write_video(self, frame):
        if self.save_video and self.video_writer is not None:
            self.video_writer.write(frame)

    def send_frame(self, gray):
        if not self.send_video or self.client_socket is None:
            return True
        try:
            ret, buffer = cv2.imencode(".png", gray)
            img_encoded = buffer.tobytes()
            data = pickle.dumps(img_encoded)
            self.client_socket.sendall(struct.pack("L", len(data)) + data)
        except Exception as e:
            print(f"Socket error: {e}")
            return False
        return True

    def release(self):
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
        if self.send_video and self.client_socket is not None:
            self.client_socket.close()

    def prepare_driving_info(self, steer_result, roi, scenario, action):
        """
        可視化用の走行情報を生成
        roi: (x1, y1, x2, y2) タプル
        steer_result: steer_by_cameraの辞書
        scenario: NormalScenarioインスタンス
        action: ActionManagerインスタンス
        """
        x1, y1, x2, y2 = roi
        mx = steer_result["mx"]
        my = steer_result["my"]
        offset_pixels = steer_result["offset_pixels"]
        max_contour = steer_result["max_contour"]
        def to_distance_cm(pos):
            return int(float(pos) * 0.0471) if pos is not None else 0
        left_distance_cm = to_distance_cm(action.left_relative_position)
        right_distance_cm = to_distance_cm(action.right_relative_position)
        color = action.color
        distance = action.distance
        left_actual = action.left_actual_power
        right_actual = action.right_actual_power
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
            "left_power": f"{action.left_power}% | {left_actual if left_actual is not None else 'N/A'}%",
            "right_power": f"{action.right_power}% | {right_actual if right_actual is not None else 'N/A'}%",
            "left_relative_position": f"{action.left_relative_position}deg / {left_distance_cm}cm",
            "right_relative_position": f"{action.right_relative_position}deg / {right_distance_cm}cm",
            "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
            "mode": scenario.mode.name
        }
        return info

    def create_visualization_frame(self, frame, steer_result, roi, info):
        """
        可視化フレームを生成し、輪郭があれば描画する
        roi: (x1, y1, x2, y2) タプル
        steer_result: steer_by_cameraの辞書
        """
        x1, y1, x2, y2 = roi
        mx = steer_result["mx"]
        my = steer_result["my"]
        max_contour = steer_result["max_contour"]
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        gray = draw_driving_info(gray, info, roi)
        if max_contour is not None:
            adjusted_contour = max_contour + np.array([x1, y1])
            cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)
            cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)
        return gray

    def process_and_send(self, frame, steer_result, roi, scenario, action):
        info = self.prepare_driving_info(steer_result, roi, scenario, action)
        gray = self.create_visualization_frame(frame, steer_result, roi, info)
        self.write_video(frame)
        return self.send_frame(gray)

def get_timestamp():
    """
    現在時刻のタイムスタンプ（YYYYMMDDHHMMSS形式）を返す共通関数。
    """
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

# --- デフォルトシナリオパターン用ベースクラス ---
class DefaultScenario:
    """
    モード遷移・アクション遷移のデフォルト基底クラス。
    すべてのシナリオクラスはこの基底クラスを継承すること。
    共通状態（modeなど）もここで初期化する。
    execute_mode_actionの引数は派生シナリオごとに異なるため、*args, **kwargsで受ける。
    """
    def __init__(self):
        self.mode = Mode.LINE_TRACE

    def transition_mode(self, *args, **kwargs):
        """
        モード遷移を行う。引数は派生クラスで適切に定義すること。
        """
        raise NotImplementedError

    def execute_mode_action(self, *args, **kwargs):
        """
        モード遷移とアクション実行を行う。引数は派生クラスで適切に定義すること。
        """
        raise NotImplementedError

# --- 通常（ノーマル）シナリオの実装例 ---
class NormalScenario(DefaultScenario):
    """
    状態遷移と動作遷移を1つのクラスで管理する通常（ノーマル）シナリオの実装例。
    """
    def __init__(self):
        self.mode = Mode.LINE_TRACE
        self.obstacle_detected_time = None

    def transition_mode(self, distance):
        OBSTACLE_DETECT_DISTANCE = 70
        DIST_STOP_DURATION = 1.0
        if self.mode == Mode.LINE_TRACE:
            if distance is not None and distance < OBSTACLE_DETECT_DISTANCE:
                self.mode = Mode.DIST_STOP
                self.obstacle_detected_time = time.time()
        elif self.mode == Mode.DIST_STOP:
            if distance is not None and distance >= OBSTACLE_DETECT_DISTANCE * (60/30):
                self.mode = Mode.LINE_TRACE
                self.obstacle_detected_time = None
            elif time.time() - self.obstacle_detected_time >= DIST_STOP_DURATION:
                self.mode = Mode.OBSTACLE_AVOID
        elif self.mode == Mode.OBSTACLE_AVOID:
            pass
        # SMART_CARRY_1, SMART_CARRY_2, GOALへの遷移は必要に応じて追加

    def execute_mode_action(self, action, steer_result):
        offset_pixels = steer_result.get("offset_pixels", 0)
        self.transition_mode(action.distance)
        if self.mode == Mode.LINE_TRACE:
            action.do_line_trace(offset_pixels)
        elif self.mode == Mode.DIST_STOP:
            action.do_dist_stop()
        elif self.mode == Mode.OBSTACLE_AVOID:
            action.do_obstacle_avoid()
            if action.is_finished():
                self.mode = Mode.LINE_TRACE  # state3後に必ずLINE_TRACEへ遷移
                self.obstacle_detected_time = None
                action.reset()  # 回避動作の状態もリセット
        elif self.mode == Mode.GOAL:
            pass
        else:
            pass

# --- マニュアルシナリオの実装例 ---
class ManualScenario(DefaultScenario):
    """
    キーボード入力による手動操作専用のシナリオ。
    DefaultScenarioを継承し、execute_mode_actionで手動制御用のロジックを実装する。
    """
    def __init__(self):
        super().__init__()
        self.mode = Mode.MANUAL
        self.manual_start_time = None

    def transition_mode(self, key=None):
        if self.mode == Mode.MANUAL:
            # Only accept key input in MANUAL mode
            if key == 'a':
                self.mode = Mode.MANUAL_A
                self.manual_a_start_time = time.time()
        elif self.mode == Mode.MANUAL_A:
            # Ignore all key input in MANUAL_A mode
            if time.time() - self.manual_a_start_time >= 1.0:
                self.mode = Mode.STOP
        elif self.mode == Mode.STOP:
            self.mode = Mode.MANUAL

    def execute_mode_action(self, action, key=None):
        self.transition_mode(key=key)
        if self.mode == Mode.MANUAL:
            pass
        elif self.mode == Mode.MANUAL_A:
            action.do_straight()  # MANUAL_Aモードで直進
        elif self.mode == Mode.STOP:
            action.do_stop()  # STOPモードで停止

# --- キーボードコントローラー（Windows/Unix両対応） ---
class KeyboardController:
    """
    キーボード入力を非ブロッキングで取得するコントローラー。
    Windows: msvcrt、Unix: termios/tty/select を利用。
    get_key()で1文字取得、何も押されていなければNone。
    """
    def __init__(self):
        self.is_windows = WINDOWS
        if not self.is_windows:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)

    def get_key(self):
        if self.is_windows:
            found_a = False
            key = None
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                print(f"[DEBUG][WIN] raw: {ch}")  # デバッグ
                if ch in (b'\x00', b'\xe0'):
                    msvcrt.getch()  # 特殊キーの2バイト目を消費
                    continue
                try:
                    decoded = ch.decode('utf-8')
                    print(f"[DEBUG][WIN] decoded: {decoded}")  # デバッグ
                    if decoded.lower() == 'a':
                        found_a = True
                    elif key is None and decoded.isprintable():
                        key = decoded.lower()
                except Exception as e:
                    print(f"[DEBUG][WIN] decode error: {e}")
                    continue
            if found_a:
                print("[DEBUG][WIN] return 'a'")
                return 'a'
            print(f"[DEBUG][WIN] return key: {key}")
            return key
        else:
            import sys, select, tty, termios
            tty.setcbreak(self.fd)
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    ch = sys.stdin.read(1)
                    print(f"[DEBUG][LINUX] ch: {repr(ch)}")
                    if ch.lower() == 'a':
                        print("[DEBUG][LINUX] return 'a'")
                        return 'a'
                    elif ch.isprintable():
                        print(f"[DEBUG][LINUX] return key: {ch.lower()}")
                        return ch.lower()
                print(f"[DEBUG][LINUX] return key: None")
                return None
            finally:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

# --- メイン処理 ---
# python run_opencv.py --record-sensor --send-video
def main(config: Config):
    # scenario = NormalScenario()
    scenario = ManualScenario()
    action = ActionManager()
    camera = Camera()
    video = VideoManager(config.log_save_video, config.log_send_video, IMAGE_WIDTH, IMAGE_HEIGHT, HOST_IP_ADDRESS, port=8485)
    sensor_recorder = SensorRecorderManager(config.log_sensor)
    action.test_initial_sensor()
    action.test_arm()
    # --- KeyboardControllerのインスタンス化（log_manualがTrueの場合のみ） ---
    if config.log_manual:
        key = KeyboardController()
    else:
        key = None
    time.sleep(0.5)
    try:
        while action.et.is_running == True:
            ret, frame = camera.read()
            if not ret:
                print("[ERROR] Can't receive frame (stream end?). Exiting ...")
                break
            action.update_sensor_info(sensor_recorder)
            if config.log_manual:
                steer_result = {"mx": 0, "my": 0, "offset_pixels": 0, "max_contour": None}
                key_input = key.get_key() if config.log_manual else None
                scenario.execute_mode_action(action, key=key_input)
            else:
                steer_result = camera.steer_by_camera(frame)
                scenario.execute_mode_action(action, steer_result)
            if (config.log_save_video or config.log_send_video) and video is not None:
                if not video.process_and_send(frame, steer_result, ROI_OPENCV, scenario, action):
                    print("[ERROR] send_camera_capture failed. Breaking main loop.")
                    break
    except KeyboardInterrupt:
        print("Interrupted by user")
        action.brake_for_duration()
    except Exception as e:
        print(f"[ERROR] Unexpected exception: {e}")
        action.brake_for_duration()
    finally:
        action.et.stop()
        camera.release()
        video.release()
        if config.log_save_video and video.video_writer is not None:
            video.video_writer.release()
            print(f"Video saved to: {video.video_filename}")
        if sensor_recorder is not None and sensor_recorder.is_enabled():
            sensor_recorder.stop()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the OpenCV-based line following robot with optional sensor recording, video saving, and PC transfer"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Record sensor data to file"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save camera video to file (on Raspberry Pi)"
    )
    parser.add_argument(
        "--send-video", action="store_true", help="Send camera video to PC via socket"
    )
    parser.add_argument(
        "--manual", action="store_true", help="Control robot with keyboard input"
    )

    args = parser.parse_args()

    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_OPENCV}")
    print(f"Base power: {BASE_POWER}")
    print("Press Ctrl+C to stop")

    config = Config(
        log_sensor=args.record_sensor,
        log_save_video=args.save_video,
        log_send_video=args.send_video,
        log_manual=args.manual
    )
    main(config)

