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
ROI_OPENCV = (150, 300, 490, 400)  # 必ずタプルで定義すること
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480                 # カメラ画像の高さ
BASE_POWER = 50                    # カーブ時の基準パワー
STRAIGHT_POWER = 80                # 直線時の推奨パワー
CURVE_POWER = 30                   # 急カーブ時の最低パワー
CURVE_THRESHOLD_DEG = 10           # カーブ判定閾値（度数法, SENSITIVITY=1.0時の推奨値）
STRAIGHT_THRESHOLD_DEG = 3         # 直線判定のしきい値（ユーザー調整用, デフォルト3度, STRAIGHT_THRESHOLD_DEGで指定）
SENSITIVITY = 1.0                  # ピクセル→theta変換感度
MAX_POWER_DIFF = 40       # 最大旋回時の左右パワー差（%）
MAX_THETA_DEG = 50        # 最大旋回角（度数法, 例: 50度）
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

    def update_and_act(self, distance, action_manager, offset_pixels=None, calc=None):
        self.update(distance)
        if self.mode == Mode.LINE_TRACE:
            return action_manager.do_line_trace(offset_pixels, calc)
        elif self.mode == Mode.DIST_STOP:
            return action_manager.do_dist_stop()
        elif self.mode == Mode.OBSTACLE_AVOID:
            return action_manager.do_obstacle_avoid()
        elif self.mode == Mode.GOAL:
            pass  # GOALモード時は何もしない（将来の拡張用）
        else:
            pass  # 未定義モードは何もしない（安全策として停止動作も行わない）
        return 0, 0, 0  # どの分岐にも該当しない場合は必ずタプルで返す


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
            Kp=2.0,
            Ki=0,
            Kd=0.4,
            setpoint=0,
            output_limits=(-0.25, 0.25),
        )
        self.et.set_motor_relative_position(left_position=0, right_position=0)

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
        return theta, pid_corrected_theta, current_power

    def do_dist_stop(self):
        self.left_power = 0
        self.right_power = 0
        self.apply_power()  # ←ここで即時モーター出力
        return 0, 0, 0

    def do_obstacle_avoid(self):
        # --- 障害物回避動作（状態遷移あり） ---
        # 1段階目: 左回転 → 2段階目: 右弧旋回 → 3段階目: 左回転
        USER_TIME_PER_DEGREE = 1.0 / 90  # ←90度で何秒かかかるか実測値で調整
        ARC_POWER = 50
        ARC_DURATION = 5.0
        TURN_ANGLE = 45
        ARC_RATIO = 0.8  # カーブ時の弱い側のパワー比
        now = time.time()
        if self.finished:
            self.left_power = 0
            self.right_power = 0
            self.apply_power()
            return 0, 0, 0
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
                    self._reset_action_vars()
                    self.finished = True
        self.apply_power()  # ←ここで即時モーター出力
        return 0, 0, 0

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

    def get_sensor_info(self, sensor_recorder=None):
        """
        Spikeの最新センサーステータス・カラー・超音波・モーター情報をまとめて取得
        - カラーセンサー値取得と黒・青判定
        - 超音波センサーデータ値取得
        - モーターA/B相対位置値・パワー値取得（A:右, B:左）
        - センサーデータ記録が有効な場合はロガーに記録
        """
        spike_status = self.et.get_spike_status()
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

    def send_stop_signal(self, duration=3.0):
        """
        Spikeに一定時間ブレーキ信号を送り続ける
        """
        print(f"Sending stop signals to Spike for {duration} seconds...")
        stop_start_time = time.time()
        while time.time() - stop_start_time < duration:
            try:
                self.et.brake()
                time.sleep(0.1)
            except Exception as e:
                print(f"Error sending stop signal: {e}")
                break
        print("Stop signal transmission completed")

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
        ret, buffer = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_encoded = buffer.tobytes()
        data = pickle.dumps(img_encoded)
        client_socket.sendall(struct.pack("L", len(data)) + data)
    except Exception as e:
        print(f"Socket error: {e}")
        return False
    return True

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

# --- メイン処理 ---
def main(record_sensor_data=False, save_camera_video=False):
    """
    メイン制御ループ
    - 各種初期化（ロボット・センサーレコーダ・ビデオ・ソケット等）
    - 初期センサーテスト・アーム動作テスト
    - メインループで以下を繰り返す：
        1. カメラ画像取得
        2. ActionManager経由でセンサー情報取得
        3. 画像処理でライン重心・オフセット検出
        4. モード遷移・制御出力（ModeManager/ActionManager）
        5. 可視化情報生成・フレーム描画
        6. 画像送信・動画保存
        7. ループ周期調整
    - 例外・割り込み時は安全停止・リソース解放
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

    # --- mainループ処理の流れ ---
    # 1. カメラ画像を取得（cap.read）
    # 2. センサー情報（カラー・超音波・モーター相対位置/パワー）をActionManager経由で取得
    # 3. （必要に応じて）モデル推論例（画像・motor_info・distanceを入力、進行方向や物体判定を出力）
    # 4. 画像処理でライン重心・オフセット・最大輪郭を検出
    # 5. モード遷移・制御出力（mode.update_and_act）
    #    - LINE_TRACE: ライントレース制御
    #    - DIST_STOP: 障害物検知時の一時停止
    #    - OBSTACLE_AVOID: 障害物回避動作
    #    - 必要に応じてGOALやSMART_CARRY等も拡張可
    # 6. モードごとの制御値をActionManagerから取得し、可視化情報を生成
    # 7. 可視化フレーム生成（輪郭・重心描画など）
    # 8. カメラ画像の送信・保存（リモート監視や動画保存）
    # 9. ループ周期制御（30ms未満ならsleepで調整）
    # 10. 例外・割り込み時は安全停止（brake/stop）・リソース解放
    #
    # ※run_opencv_bk0626.pyの設計例を参考に、OOP設計・責務分担を明確化
    #
    # 各処理はActionManager/ModeManager/ControlCalculator等の責務に分離し、
    # mainは「全体の流れ・状態遷移・例外処理・リソース管理」のみを記述
    #
    # モデル推論例や拡張例はコメント参照

    try:
        while action.et.is_running == True:
            loop_start = time.time()
            # 1. カメラ画像取得
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Can't receive frame (stream end?). Exiting ...")
                break
            # 2. センサー情報取得（ActionManager経由で一括取得）
            color, distance, motor_info = action.get_sensor_info(sensor_recorder)
            # 3. （必要に応じて）モデル推論例（コメント参照）
            # 4. 画像処理でライン重心・オフセット・最大輪郭を検出
            mx, my, offset_pixels, max_contour = calc.steer_by_camera(frame)
            # 5. モード遷移・制御出力（ModeManager/ActionManager）
            theta, pid_corrected_theta, current_power = mode.update_and_act(
                distance,
                action,
                offset_pixels=offset_pixels,
                calc=calc
            )
            # OBSTACLE_AVOIDモード終了時はモードリセット
            if mode.mode == Mode.OBSTACLE_AVOID:
                if action.is_finished():
                    mode.reset()
            left_power = action.left_power
            right_power = action.right_power
            # 6. 可視化情報生成
            info = prepare_driving_info(ROI_OPENCV, mx, my, offset_pixels, theta, pid_corrected_theta, current_power, left_power, right_power, color, distance, motor_info, max_contour, mode=mode.mode.name)
            # 7. 可視化フレーム生成
            gray = create_visualization_frame(frame, info, ROI_OPENCV, mx, my, max_contour)
            # 8. カメラ画像の送信・保存
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)
            if not send_camera_capture(gray, client_socket):
                print("[ERROR] send_camera_capture failed. Breaking main loop.")
                break
            # 9. ループ周期調整（30ms未満ならsleep）
            elapsed = time.time() - loop_start
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)
    except KeyboardInterrupt:
        # ユーザーによる割り込み（Ctrl+C）時：安全のため一定時間ブレーキ信号を連続送信
        print("Interrupted by user")
        action.send_stop_signal()  # Spikeに3秒間ブレーキ信号を送り続ける
    except Exception as e:
        # 予期しない例外発生時も必ずロボットを安全に停止（3秒間ブレーキ信号送信）し、例外内容を表示
        print(f"[ERROR] Unexpected exception: {e}")
        action.send_stop_signal()  # Spikeに3秒間ブレーキ信号を送り続ける
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
