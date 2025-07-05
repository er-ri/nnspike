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

# 標準ライブラリ
import argparse
import math
import pickle
import socket
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum, auto

# サードパーティライブラリ
import cv2
import numpy as np

# プラットフォーム固有のインポート
try:
    import msvcrt
    WINDOWS = True
except ImportError:
    import termios
    import tty
    import select
    WINDOWS = False

# プロジェクト固有のインポート
from nnspike.unit import ETRobot
from nnspike.utils import (
    SensorRecorder,
    PIDController,
    ControlCalculator,
    Camera,
)

# ==== ユーザー調整用パラメータ（ここだけ編集すればOK） ====
 # ROI_OPENCV: OpenCV画像処理で使用する領域（左上x, 左上y, 右下x, 右下y）
ROI_BOTTLE = (100, 20, 540, 450)   # (x1, y1, x2, y2)  # 下端を50上げて下側を広げる
# ROI_OPENCV: 左右を均等に30ピクセルずつ広げる（例: x1-30, x2+30）
ROI_OPENCV = (10, 50, 630, 450)   # x1=40-30=10, x2=600+30=630, 左右さらに30pxずつ拡張
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480                 # カメラ画像の高さ
# 黒判定の閾値（反射光R: 40以下, color: 150以下なら黒と判定）
BLACK_REFLECTED_THRESHOLD = 40
BLACK_COLOR_THRESHOLD = 150
# 青判定の閾値（色値: 30以下, 反射光: 60以下なら青と判定）
BLUE_COLOR_THRESHOLD = 30
BLUE_REFLECTED_THRESHOLD = 60
BOTTLE_YELLOW_THRESHOLD = 12000
BOTTLE_BLUE_THRESHOLD = 14000
BOTTLE_RED_THRESHOLD = 14000
MOTOR_SEND_INTERVAL = 0.04

POSITION_YELLOW_BOTTLE = 3000         # 黄色ボトル回避開始位置
POSITION_BLUE_BOTTLE = 20000           # 青ボトル運搬開始位置
POSITION_RED_BOTTLE = 250000            # 赤ボトル運搬開始位置
# ================================================

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
    OBSTACLE_AVOID = auto()     # 障害物回避動作（没）
    SMART_CARRY_1 = auto()      # スマートキャリー1回目
    SMART_CARRY_2 = auto()      # スマートキャリー2回目
    GOAL = auto()               # ゴール到達モード
    MANUAL = auto()             # マニュアル操作モード（入力待ち）
    STOP = auto()               # 停止モード
    MANUAL_A = auto()           # マニュアル直進モード
    MANUAL_B = auto()           # マニュアル障害物回避モード
    MANUAL_D = auto()           # マニュアル直進＋ボトル検出モード
    MANUAL_E = auto()           # マニュアルEモード（新規追加）
    YELLOW_BOTTLE = auto()      # 黄色ボトル検出時モード
    BLUE_BOTTLE = auto()        # 青ボトル検出時モード
    RED_BOTTLE = auto()         # 赤ボトル検出時モード

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
        self.pid = PIDController()
        self.calc = ControlCalculator(IMAGE_WIDTH, IMAGE_HEIGHT, roi_opencv=ROI_OPENCV)
        self.et.set_motor_relative_position(left_position=0, right_position=0)
        self.reset_control_values()  # theta, pid_corrected_theta, current_powerをまとめてリセット
        # --- 受信した実際の左右パワー値を初期化 ---
        self.left_actual_power = None
        self.right_actual_power = None
        self.last_send_time = None  # 送信タイムスタンプ（ms差分計算用）
        self.is_stopped = False  # STOP状態フラグを追加

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

    def _apply_power_common(self, forward=True, force_immediate=False):
        """forward=True: set_motor_forward_power, False: set_motor_backward_power
        force_immediate=Trueの場合は送信間隔待ちをスキップし即時送信"""
        # --- 送信間隔をユーザー調整パラメータ MOTOR_SEND_INTERVAL で制御 ---
        now = time.time()
        if not force_immediate and hasattr(self, 'last_send_time') and self.last_send_time is not None:
            elapsed = now - self.last_send_time
            wait = MOTOR_SEND_INTERVAL - elapsed
            while wait > 0:
                time.sleep(min(wait, 0.001))
                now = time.time()
                elapsed = now - self.last_send_time
                wait = MOTOR_SEND_INTERVAL - elapsed
        # --- ここから送信処理 ---
        if forward:
            self.et.set_motor_forward_power(left_power=self.left_power, right_power=self.right_power)
        else:
            self.et.set_motor_backward_power(left_power=self.left_power, right_power=self.right_power)
        # spikeから実際の左右パワーを取得
        status = self.et.get_spike_status()
        left_actual = None
        right_actual = None
        if status and hasattr(status, 'motors'):
            left_actual = status.motors['B'].power if 'B' in status.motors and hasattr(status.motors['B'], 'power') else None
            right_actual = status.motors['A'].power if 'A' in status.motors and hasattr(status.motors['A'], 'power') else None
        self.left_actual_power = left_actual
        self.right_actual_power = right_actual
        # --- ms単位で送信タイミングと前回からの差分をデバッグ出力 ---
        now = time.time()
        ts = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
        ms = int((now - int(now)) * 1000)
        if hasattr(self, 'last_send_time') and self.last_send_time is not None:
            diff_ms = int((now - self.last_send_time) * 1000)
            label = "apply_power" if forward else "apply_power_backward"
            # print(f"[DEBUG][{label}] sent at {ts}.{ms:03d} (+{diff_ms}ms)")
        else:
            label = "apply_power" if forward else "apply_power_backward"
            # print(f"[DEBUG][{label}] sent at {ts}.{ms:03d} (first)")
        self.last_send_time = now

    def apply_power(self):
        """現在のleft_power, right_powerをロボットに反映し、spikeから実際の左右パワーを取得（前進）"""
        self._apply_power_common(forward=True)

    def apply_power_backward(self):
        """現在のleft_power, right_powerをロボットに反映（バック用: set_motor_backward_power使用）、spikeから実際の左右パワーを取得"""
        self._apply_power_common(forward=False)

    def apply_power_immediate(self):
        """送信間隔待ちをせず即時に現在のパワー値を送信（停止命令用）"""
        self._apply_power_common(forward=True, force_immediate=True)

    def do_stop(self):
        self.left_power = 0
        self.right_power = 0
        self.apply_power_immediate()
        self.is_stopped = True  # STOP状態にセット

    def do_straight(self):
        self.left_power = 50
        self.right_power = 50
        self.apply_power()

    def brake_for_duration(self, duration=10.0):
        """
        Spike本体に一定時間（デフォルト10秒）ブレーキ信号を連続送信し、安全停止を強制する
        インターバルは50ms固定
        """
        print(f"[SAFETY] Sending brake command to Spike for {duration} seconds (brake_for_duration)")
        stop_start_time = time.time()
        interval = 0.05  # 50ms
        count = 0
        while time.time() - stop_start_time < duration:
            try:
                self.left_power = 0
                self.right_power = 0
                self.apply_power_immediate()
                self.et.brake()
                count += 1
                time.sleep(interval)
            except Exception as e:
                print(f"[SAFETY][ERROR] Exception during brake command: {e}")
                break
        print(f"[SAFETY] Brake command transmission completed (brake_for_duration), total sends: {count}")

    def reset_control_values(self):
        self.theta = 0
        self.pid_corrected_theta = 0
        self.current_power = 0
        self.is_stopped = False  # STOP状態解除

    def do_line_trace(self, offset_pixels, position=None):
        if getattr(self, 'is_stopped', False):
            return  # STOP状態なら何もしない
        # === ライントレース制御のメイン処理 ===
        # 1. 進行角度thetaをオフセットピクセルから算出
        theta = self.calc.calculate_attitude_angle(offset_pixels)
        # 2. θとpositionに応じた推奨速度（パワー）を決定
        if position is None:
            position = getattr(self, 'right_relative_position', 0)
        current_power = self.calc.calculate_adaptive_speed(theta, position)
        # 3. PID制御で進行角度を補正
        pid_corrected_theta = self.pid.update(theta)
        # 4. PID補正値をパワー差分に変換
        power_adjustment = self.calc.calculate_power_adjustment(pid_corrected_theta)
        # 5. 左右パワーを計算（負値にならないようクリッピング）
        self.left_power = max(0, int(current_power - power_adjustment))
        self.right_power = max(0, int(current_power + power_adjustment))
        # 6. モーター出力を即時反映
        self.apply_power()
        # 7. デバッグ・可視化用の値を保存
        self.theta = theta
        self.pid_corrected_theta = pid_corrected_theta
        self.current_power = current_power

    def do_obstacle_avoid(self):
        if getattr(self, 'is_stopped', False):
            return
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
        self.apply_power()  # ←ここで即時モーター出力

    def do_obstacle_avoid_no_bottle(self):
        if getattr(self, 'is_stopped', False):
            return
        # --- 障害物回避動作（状態遷移あり, 8段階） ---
        now = time.time()
        if self.finished:
            self.left_power = 0
            self.right_power = 0
            self.apply_power()
            return
        if self.state == 0:
            # 直進
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + 2.0
                self.left_power = 80
                self.right_power = 80
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 1
                    self._reset_action_vars()
        elif self.state == 1:
            # 45度右旋回
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.turn_duration = 45 * 1.0 / 90
                self.left_power = 0
                self.right_power = 30
            else:
                if now - self.start_time >= self.turn_duration:
                    self.et.brake()
                    self.state = 2
                    self._reset_action_vars()
        elif self.state == 2:
            # 直進
            if not self.action_sent:  
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + 1.0
                self.left_power = 80
                self.right_power = 80
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 3
                    self._reset_action_vars()
        elif self.state == 3:
            # 90度左旋回
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.turn_duration = 90 * 1.0 / 90
                self.left_power = 30
                self.right_power = 0
            else:
                if now - self.start_time >= self.turn_duration:
                    self.et.brake()
                    self.state = 4
                    self._reset_action_vars()
        elif self.state == 4:
            # 直進
            if not self.action_sent:  
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + 1.0
                self.left_power = 80
                self.right_power = 80
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 5
                    self._reset_action_vars()
        elif self.state == 5:
            # 45度右旋回
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.turn_duration = 45 * 1.0 / 90
                self.left_power = 0
                self.right_power = 30
            else:
                if now - self.start_time >= self.turn_duration:
                    self.et.brake()
                    self.state = 6
                    self._reset_action_vars()
        elif self.state == 6:
            # 直進
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + 2.0
                self.left_power = 80
                self.right_power = 80
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 7
                    self._reset_action_vars()
        elif self.state == 7:
            # 完了
            self.finished = True
        self.apply_power()
        
    def do_obstacle_avoid_with_bottle(self):
        if getattr(self, 'is_stopped', False):
            return
        # --- 障害物回避動作（3段階：右向き→左迂回→完了） ---
        now = time.time()
        if self.finished:
            self.left_power = 0
            self.right_power = 0
            self.apply_power()
            return
        if self.state == 0:
            # 0.5秒間右を強くして左向き（両モーター動作、右が強い）
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + 1
                self.left_power = 20  # 左モーターも動かす（弱く）
                self.right_power = 60 # 右モーターを強く（左向き）
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 1
                    self._reset_action_vars()
        elif self.state == 1:
            # 2秒間左を強くして迂回
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.arc_end_time = now + 2.0
                self.left_power = 60  # 左モーターを強く
                self.right_power = 30 # 右モーターを弱く（左迂回）
            else:
                if now >= self.arc_end_time:
                    self.et.brake()
                    self.state = 2
                    self._reset_action_vars()
        elif self.state == 2:
            # 完了
            self.finished = True
        self.apply_power()

    def carry_bottle_sequence(self):
        if getattr(self, 'is_stopped', False):
            return
        # --- キャリーボトル運搬動作（state=0から開始、直進→停止→運搬→完了） ---
        now = time.time()
        if self.finished:
            self.left_power = 0
            self.right_power = 0
            self.apply_power()
            return
        if self.state == 0:
            # ゆっくり直進してボトルキャッチ
            if not self.action_sent:  
                self.start_time = now
                self.action_sent = True
                self.end_time = now + 2.0
                self.left_power = 10
                self.right_power = 10
            else:
                if now >= self.end_time:
                    self.et.brake()
                    self.state = 1
                    self._reset_action_vars()
        elif self.state == 1:
            # 1秒間停止
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.left_power = 0
                self.right_power = 0
            else:
                if now - self.start_time >= 1.0:
                    self.et.brake()
                    self.state = 2
                    self._reset_action_vars()
        elif self.state == 2:
            # 2秒間power30で運ぶ
            if not self.action_sent:  
                self.start_time = now
                self.action_sent = True
                self.end_time = now + 2.0
                self.left_power = 30
                self.right_power = 30
            else:
                if now >= self.end_time:
                    self.et.brake()
                    self.state = 3
                    self._reset_action_vars()
        elif self.state == 3:
            # 2秒間バック
            if not self.action_sent:
                self.start_time = now
                self.action_sent = True
                self.end_time = now + 2.0
                self.left_power = 30   # バック時もプラス値でOK（apply_power_backwardで方向制御）
                self.right_power = 30
            else:
                if now >= self.end_time:
                    self.et.brake()
                    self.state = 4
                    self._reset_action_vars()
            self.apply_power_backward()  # バック時はapply_power_backwardのみ呼ぶ
            return
        elif self.state == 4:
            # 完了フラグのみ
            self.finished = True
        self.apply_power()

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

    def update_sensor_state(self):
        """
        Spikeの最新センサーステータス・カラー・超音波・モーター情報をまとめて取得し、インスタンス変数に格納（レコーダー出力はしない）
        """
        spike_status = self.et.get_spike_status()
        sensors = spike_status.sensors
        self._latest_spike_status = spike_status  # レコーダー出力用に保持
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

    def log_sensor_record(self, sensor_recorder=None, bottle_result=None, steer_result=None):
        """
        最新のspike_status, bottle_result, steer_resultをレコーダーに記録（self._latest_spike_statusを利用）
        """
        if not hasattr(self, '_latest_spike_status'):
            return
        spike_status = self._latest_spike_status
        if sensor_recorder is not None and sensor_recorder.is_enabled():
            try:
                sensor_recorder.recorder.log_frame_data(spike_status, bottle_result=bottle_result, steer_result=steer_result)
            except Exception as e:
                print(f"[SensorRecorderManager] log error: {e}")
                traceback.print_exc()

    def brake_for_duration(self, duration=3.0):
        """
        Spike本体に一定時間、左右パワーを0にするだけの「緩い停止命令」を送信する（brakeやstopは呼ばない）
        - 途中でKeyboardInterruptを受け付け、即座にbreak
        - 送信間隔はMOTOR_SEND_INTERVALに合わせる
        - 送信失敗時もbreak
        """
        print(f"[SAFETY] Sending gentle stop command to Spike for {duration} seconds (brake_for_duration)")
        stop_start_time = time.time()
        count = 0
        try:
            while time.time() - stop_start_time < duration:
                self.left_power = 0
                self.right_power = 0
                self.apply_power_immediate()
                count += 1
                time.sleep(MOTOR_SEND_INTERVAL)
        except KeyboardInterrupt:
            print("[SAFETY] KeyboardInterrupt during gentle stop. Exiting loop.")
        except Exception as e:
            print(f"[SAFETY][ERROR] Exception during gentle stop command: {e}")
        print(f"[SAFETY] Gentle stop command transmission completed (brake_for_duration), total sends: {count}")

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
            # --- サーバが起動するまで2秒ごとにリトライ（最大30秒でタイムアウト） ---
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            start_time = time.time()
            while True:
                try:
                    self.client_socket.connect((host_ip_address, port))
                    print(f"[VideoManager] Connected to server at {host_ip_address}:{port}")
                    break
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed >= 30:
                        print(f"[VideoManager] Connection timeout after 30 seconds. Could not connect to server at {host_ip_address}:{port}.")
                        self.client_socket = None
                        break
                    print(f"[VideoManager] Waiting for server at {host_ip_address}:{port}... ({e})")
                    time.sleep(2)

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

    def prepare_driving_info(self, steer_result, scenario, action, bottle=None):
        """
        可視化用の走行情報を生成
        steer_result: steer_by_cameraの辞書
        scenario: NormalScenarioインスタンス
        action: ActionManagerインスタンス
        bottle: ペットボトル検出結果（色ごとのピクセル数辞書など）
        """
        # ROIは現状opencvのみを使用
        x1, y1, x2, y2 = ROI_OPENCV
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
        # --- bottle情報を色ごとのピクセル数で表示 ---
        pixel_dict = bottle[0] if (bottle and isinstance(bottle, tuple)) else (bottle if isinstance(bottle, dict) else {})
        if pixel_dict:
            yellow = pixel_dict.get('yellow', 0) or 0
            blue = pixel_dict.get('blue', 0) or 0 
            red = pixel_dict.get('red', 0) or 0
            bottle_str = f"Y:{yellow} B:{blue} R:{red}"
        else:
            bottle_str = "N/A"
        info = dict()
        # ROIは現状opencvのみを使用
        info["offset_x"], info["offset_y"] = int(x1 + mx), int(y1 + my)
        info["roi"] = ROI_OPENCV
        # 表示用の数値処理（最小限の変換のみ）
        left_power = action.left_power or 0
        right_power = action.right_power or 0
        left_actual_val = left_actual if left_actual is not None else 'N/A'
        right_actual_val = right_actual if right_actual is not None else 'N/A'
        left_rel_pos = action.left_relative_position or 0
        right_rel_pos = action.right_relative_position or 0
        left_distance_cm_val = left_distance_cm or 0
        right_distance_cm_val = right_distance_cm or 0
        # スピード表示・power_statusは常に表示（条件分岐なし）
        current_power_str = f"{round(action.current_power, 1)}%"
        power_status_str = "CURVE"
        info["text"] = {
            "offset_pixels": f"{round(offset_pixels, 1)}px",
            "theta_deg": f"{round(math.degrees(action.theta), 2)}deg",
            "pid_corrected_theta": f"{round(math.degrees(action.pid_corrected_theta), 2)}deg",
            "power_status": power_status_str,
            "current_power": current_power_str,
            "on_color": (
                "BLACK" if (color and color.is_black) else ("BLUE" if (color and color.is_blue) else "N/A")
            ),
            "color_sensor": color_data,
            "ultrasonic_sensor": ultrasonic_data,
            "left_power": f"{left_power}% | {-left_actual_val if left_actual != None else 'N/A'}%",
            "right_power": f"{right_power}% | {right_actual_val if right_actual != None else 'N/A'}%",
            "left_relative_position": f"{left_rel_pos}deg / {left_distance_cm_val}cm",
            "right_relative_position": f"{right_rel_pos}deg / {right_distance_cm_val}cm",
            "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
            "mode": scenario.mode.name,
            "bottle": bottle_str
        }
        return info

    def create_visualization_frame(self, frame, steer_result, info, bottle_masks=None):
        """
        可視化フレームを生成する関数。

        - カメラから取得したRGB画像（frame）をグレースケール画像に変換。
        - draw_driving_info関数を呼び出し、ROI矩形や走行情報（テキスト、オフセット点など）を重畳。
        - 必要に応じて最大輪郭や重心点も描画。
        - 生成した可視化フレーム（gray）を返す。

        Args:
            frame (np.ndarray): カメラから取得したRGB画像。
            steer_result (dict): 画像処理結果（重心座標、最大輪郭など）。
            info (dict): 走行情報（draw_driving_info用）。
            bottle_masks (dict): ROI_BOTTLE内の各色マスク画像（オプション）。
        Returns:
            np.ndarray: 可視化情報が重畳されたグレースケール画像。
        """
        # ROIは現状opencvのみを使用
        x1, y1, x2, y2 = ROI_OPENCV
        mx = steer_result["mx"]
        my = steer_result["my"]
        max_contour = steer_result["max_contour"]
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        # draw_driving_infoでROI_OPENCV（赤）とROI_BOTTLE（緑）を必ず両方描画するため、ここでは描画しない
        gray = draw_driving_info(gray, info)
        if max_contour is not None:
            adjusted_contour = max_contour + np.array([x1, y1])
            if len(max_contour) == 2:  # 線分の場合
                pt1 = tuple(adjusted_contour[0][0])
                pt2 = tuple(adjusted_contour[1][0])
                cv2.line(gray, pt1, pt2, (255, 255, 255), 3)  # 白線で描画
            else:  # 輪郭の場合
                cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)
            # 重心点も描画
            cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)

        # --- ROI_BOTTLE内のカラーエリア（各色マスク）を白線で描画 ---
        if bottle_masks is not None:
            bx1, by1, _, _ = ROI_BOTTLE
            for color in ["yellow", "blue", "red"]:
                mask = bottle_masks.get(color)
                if mask is not None:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        adjusted_cnt = cnt + np.array([[[bx1, by1]]])
                        cv2.drawContours(gray, [adjusted_cnt], -1, (255, 255, 255), 2)
        return gray

    def process_and_send(self, frame, steer_result, scenario, action, bottle_result=None, bottle_masks=None):
        # 走行情報を準備し、可視化フレームを生成してビデオ保存・送信
        info = self.prepare_driving_info(steer_result, scenario, action, bottle=bottle_result)
        gray = self.create_visualization_frame(frame, steer_result, info, bottle_masks=bottle_masks)
        self.write_video(frame)
        return self.send_frame(gray)

def draw_driving_info(
    image: np.ndarray, info: dict
) -> np.ndarray:
    """画像上に走行情報を描画する関数。

    この関数は、与えられた画像に走行関連情報（オフセット点、ROI矩形、各種テキスト情報）をinfo辞書に基づいて描画します。

    引数:
        image (np.ndarray): 情報を描画する入力画像。
        info (dict): 表示する走行情報を含む辞書。
            期待されるキー:
                - "offset_x" (int): オフセット点のx座標。
                - "offset_y" (int): オフセット点のy座標。
                - "roi" (tuple): ROI矩形 (x1, y1, x2, y2)。
                - "text" (dict): ラベルをキー、表示内容を値とするテキスト情報の辞書。
    戻り値:
        np.ndarray: 走行情報が重畳された画像。
    """
    offset_x, offset_y = int(info["offset_x"]), int(info["offset_y"])
    x1, y1, x2, y2 = info["roi"]  # info辞書からROIを取得


    # ROI_OPENCV（赤）とROI_BOTTLE（緑）を必ず両方描画
    image = cv2.rectangle(image, ROI_OPENCV[:2], ROI_OPENCV[2:], (0, 0, 255), 2)  # 赤
    image = cv2.rectangle(image, ROI_BOTTLE[:2], ROI_BOTTLE[2:], (0, 255, 0), 2)  # 緑


    # OFFSET_Yのライン線を描画（camera.py/control.pyと同じ値を使う）
    try:
        from nnspike.utils.camera import OFFSET_Y
    except ImportError:
        OFFSET_Y = 380  # fallback
    image = cv2.line(image, (ROI_OPENCV[0], OFFSET_Y), (ROI_OPENCV[2], OFFSET_Y), (0, 255, 255), 2)  # 黄色

    image = cv2.circle(
        image, (offset_x, offset_y), 3, (255, 255, 0), -1
    )  # トレース点
    # もともとのROI（info["roi"]）も強調したい場合はここで色を変えて重ね描きも可
    # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 例: 青で細く重ね描き

    for index, key in enumerate(info["text"]):
        value = info["text"][key]
        text = f"{value:.2f}" if type(value) is float else value

        image = cv2.putText(
            image,
            f"{key} : {text}",
            (10, 30 + index * 20),  # 左上(10,30)から縦に並べる
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_4,
        )

    return image

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
        self.bottle = None
        self.offset_pixels = 0
        self.key = None
        self.start_time = None  # manual_start_time → start_time に統一

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
    【ノーマルシナリオ】
    - 自動走行（ライン検出→ボトル検出→回避/運搬）を一括管理。
    - 状態（mode）に応じてアクションを切り替え、状態遷移も自動で行う。
    - 各分岐・遷移条件はボトル検出ピクセル数やアクション完了フラグで判定。
    """
    def __init__(self):
        super().__init__()
        self.mode = Mode.LINE_TRACE  # ノーマルシナリオは常にLINE_TRACEから開始
        # ノーマルシナリオ固有の初期化があればここに追加

    def transition_mode(self, action=None):
        # --- 状態遷移: ボトル検出ピクセル数やアクションのモーター相対位置に応じて分岐 ---
        pixel_dict = self.bottle[0] if (self.bottle and isinstance(self.bottle, tuple)) else (self.bottle if isinstance(self.bottle, dict) else {})
        yellow_pixels = pixel_dict.get('yellow', 0)
        blue_pixels = pixel_dict.get('blue', 0)
        red_pixels = pixel_dict.get('red', 0)
        prev_mode = self.mode
        # --- actionのposition情報取得（Noneの場合は0に） ---
        right_pos = getattr(action, 'right_relative_position', 0) if action is not None else 0
        # --- 例: ある相対位置を超えたらSTOPやGOALに遷移するなどの拡張が可能 ---
        # ここでは例として、左右どちらかの相対位置が+/-1000度を超えたらSTOPに遷移する例を追加
        # --- モーター相対位置によるシナリオ遷移用の指標（deg単位） ---
        if self.mode == Mode.LINE_TRACE:
            # ライントレース中に各色ボトルを検出したら該当モードへ遷移
            if yellow_pixels >= BOTTLE_YELLOW_THRESHOLD and abs(right_pos) <= POSITION_YELLOW_BOTTLE:
                self.mode = Mode.YELLOW_BOTTLE
            elif blue_pixels >= BOTTLE_BLUE_THRESHOLD and abs(right_pos) >= POSITION_BLUE_BOTTLE:
                self.mode = Mode.BLUE_BOTTLE
            elif red_pixels >= BOTTLE_RED_THRESHOLD and abs(right_pos) >= POSITION_RED_BOTTLE:
                self.mode = Mode.RED_BOTTLE
        elif self.mode == Mode.YELLOW_BOTTLE:
            # 黄色ボトル回避中（完了はアクション側で判定）
            pass
        elif self.mode == Mode.BLUE_BOTTLE:
            # 青ボトル運搬中
            pass
        elif self.mode == Mode.RED_BOTTLE:
            # 赤ボトル運搬中
            pass
        elif self.mode == Mode.STOP:
            # 停止状態（未実装）
            self.mode = Mode.STOP
        # SMART_CARRY_1, SMART_CARRY_2, GOAL等は必要に応じて追加
        # --- LINE_TRACE→他モード遷移時に一度だけリセット ---
        if hasattr(self, 'prev_mode'):
            prev_mode = self.prev_mode
        else:
            prev_mode = None
        if prev_mode == Mode.LINE_TRACE and self.mode != Mode.LINE_TRACE and action is not None:
            action.reset_control_values()
        self.prev_mode = self.mode

    def execute_mode_action(self, action, steer_result, bottle=None):
        self.bottle = bottle
        self.offset_pixels = 0
        if steer_result is not None:
            self.offset_pixels = steer_result.get("offset_pixels", 0)
        self.transition_mode(action=action)
        offset_pixels = self.offset_pixels
        position = getattr(action, 'right_relative_position', None)
        if self.mode == Mode.LINE_TRACE:
            action.do_line_trace(offset_pixels, position=position)
        elif self.mode == Mode.YELLOW_BOTTLE:
            action.do_obstacle_avoid_with_bottle()
            if action.is_finished():
                self.mode = Mode.LINE_TRACE
                action.reset()
        elif self.mode == Mode.BLUE_BOTTLE or self.mode == Mode.RED_BOTTLE:
            action.carry_bottle_sequence()
            if action.is_finished():
                self.mode = Mode.STOP
                action.reset()
        elif self.mode == Mode.STOP:
            action.do_stop()
        elif self.mode == Mode.GOAL:
            pass
        else:
            pass

class ManualScenario(DefaultScenario):
    """
    【マニュアルシナリオ】
    - キーボード操作による手動制御。
    - MANUAL_Eモードではライン追従も可能。
    - 状態（mode）やキー入力に応じてアクションを切り替える。
    """
    def __init__(self):
        super().__init__()
        self.mode = Mode.MANUAL  # マニュアルシナリオは常にMANUALから開始
        # マニュアルシナリオ固有の初期化があればここに追加

    def transition_mode(self, action=None):
        pixel_dict = self.bottle[0] if (self.bottle and isinstance(self.bottle, tuple)) else (self.bottle if isinstance(self.bottle, dict) else {})
        key = self.key
        if self.mode == Mode.MANUAL:
            # a/b/d/eキーで手動モード遷移
            if key == 'a':
                self.mode = Mode.MANUAL_A
                self.start_time = time.time()
            elif key == 'b':
                self.mode = Mode.MANUAL_B
                self.start_time = time.time()
            elif key == 'd':
                self.mode = Mode.MANUAL_D
                self.start_time = time.time()
            elif key == 'e':
                self.mode = Mode.MANUAL_E
                self.start_time = time.time()
        elif self.mode == Mode.MANUAL_A:
            # MANUAL_Aは1秒経過でSTOP
            if time.time() - self.start_time >= 1.0:
                self.mode = Mode.STOP
        elif self.mode == Mode.MANUAL_B:
            # MANUAL_Bはアクション完了でSTOP（アクション側で判定）
            pass
        elif self.mode == Mode.MANUAL_D or self.mode == Mode.MANUAL_E:
            # MANUAL_D/E中にボトル検出で自動遷移
            yellow_pixels = pixel_dict.get('yellow', 0)
            blue_pixels = pixel_dict.get('blue', 0)
            red_pixels = pixel_dict.get('red', 0)
            if yellow_pixels >= BOTTLE_YELLOW_THRESHOLD:
                self.mode = Mode.YELLOW_BOTTLE
            elif blue_pixels >= BOTTLE_BLUE_THRESHOLD:
                self.mode = Mode.BLUE_BOTTLE
            elif red_pixels >= BOTTLE_RED_THRESHOLD:
                self.mode = Mode.RED_BOTTLE
        elif self.mode == Mode.YELLOW_BOTTLE:
            # 黄色ボトル回避中
            pass
        elif self.mode == Mode.BLUE_BOTTLE:
            # 青ボトル運搬中
            pass
        elif self.mode == Mode.RED_BOTTLE:
            # 赤ボトル運搬中
            pass
        elif self.mode == Mode.STOP:
            # 停止後はMANUALに復帰
            self.mode = Mode.MANUAL
        # --- MANUAL_E→他モード遷移時に一度だけリセット ---
        if hasattr(self, 'prev_mode'):
            prev_mode = self.prev_mode
        else:
            prev_mode = None
        if prev_mode == Mode.MANUAL_E and self.mode != Mode.MANUAL_E and action is not None:
            action.reset_control_values()
        self.prev_mode = self.mode

    def execute_mode_action(self, action, steer_result=None, bottle=None, key=None):
        self.bottle = bottle
        self.key = key
        if steer_result is not None:
            self.offset_pixels = steer_result.get("offset_pixels", 0)
        self.transition_mode(action=action)
        offset_pixels = self.offset_pixels
        position = getattr(action, 'right_relative_position', None)
        if self.mode == Mode.MANUAL:
            pass
        elif self.mode == Mode.MANUAL_A:
            action.do_straight()
        elif self.mode == Mode.MANUAL_B:
            action.do_obstacle_avoid_no_bottle()
            if action.is_finished():
                self.mode = Mode.STOP
                action.reset()
        elif self.mode == Mode.MANUAL_D:
            action.do_straight()
        elif self.mode == Mode.MANUAL_E:
            action.do_line_trace(offset_pixels, position=position)
        elif self.mode == Mode.YELLOW_BOTTLE:
            action.do_obstacle_avoid_with_bottle()
            if action.is_finished():
                self.mode = Mode.STOP
                action.reset()
        elif self.mode == Mode.BLUE_BOTTLE or self.mode == Mode.RED_BOTTLE:
            action.carry_bottle_sequence()
            if action.is_finished():
                self.mode = Mode.STOP
                action.reset()
        elif self.mode == Mode.STOP:
            action.do_stop()

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
            found_b = False
            found_d = False
            found_e = False
            key = None
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'\x00', b'\xe0'):
                    msvcrt.getch()  # 特殊キーの2バイト目を消費
                    continue
                try:
                    decoded = ch.decode('utf-8')
                    if decoded.lower() == 'a':
                        found_a = True
                    elif decoded.lower() == 'b':
                        found_b = True
                    elif decoded.lower() == 'd':
                        found_d = True
                    elif decoded.lower() == 'e':
                        found_e = True
                    elif key is None and decoded.isprintable():
                        key = decoded.lower()
                except Exception as e:
                    continue
            if found_a:
                return 'a'
            if found_b:
                return 'b'
            if found_d:
                return 'd'
            if found_e:
                return 'e'
            return key
        else:
            tty.setcbreak(self.fd)
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.01)  # タイムアウトを0.01秒に
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 'a':
                        return 'a'
                    elif ch.lower() == 'b':
                        return 'b'
                    elif ch.lower() == 'd':
                        return 'd'
                    elif ch.lower() == 'e':
                        return 'e'
                    elif ch.isprintable():
                        return ch.lower()
                return None
            finally:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

# --- メイン処理 ---
# python run_opencv.py --record-sensor --send-video
def main(config: Config):
    # --- シナリオ・コントローラ初期化 ---
    if config.log_manual:
        scenario = ManualScenario()
        key = KeyboardController()
    else:
        scenario = NormalScenario()
        key = None
    action = ActionManager()
    camera = Camera(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=30, roi=ROI_OPENCV, roi_bottle=ROI_BOTTLE)
    video = VideoManager(config.log_save_video, config.log_send_video, IMAGE_WIDTH, IMAGE_HEIGHT, HOST_IP_ADDRESS, port=8485)
    sensor_recorder = SensorRecorderManager(config.log_sensor)
    calc = ControlCalculator(IMAGE_WIDTH, IMAGE_HEIGHT, roi_opencv=ROI_OPENCV)
    action.test_initial_sensor()
    action.test_arm()
    time.sleep(0.5)
    try:
        while action.et.is_running == True:
            loop_start = time.time()
            ret, frame = camera.read()
            if not ret:
                print("[ERROR] Can't receive frame (stream end?). Exiting ...")
                break

            # === メイン制御ループ ===
            # 1. 画像取得（frame）
            # 2. センサー情報を最新化（self.xxx更新のみ、レコーダー出力はしない）
            action.update_sensor_state()
            # 3. ラインエッジ検出（left_x, right_x, line_width）
            left_x, right_x, line_width = camera.get_line_edges_at_y(frame)
            # 4. ステアリング計算（steer_result: ライントレース用画像処理結果）
            steer_result = calc.calc_steer_result(left_x, right_x, position=action.right_relative_position)
            # 5. ペットボトル検出（bottle_result: 色ごとのピクセル数辞書, bottle_masks: 各色マスク）
            bottle_result, bottle_masks = camera.detect_color_bottle(frame)  # 辞書とマスク両方を取得
            # 6. センサ情報・CSV記録（steer_result, bottle_resultを計算後に記録）
            action.log_sensor_record(sensor_recorder, bottle_result=bottle_result, steer_result=steer_result)
            # 7. キー入力取得（マニュアル時のみ）
            key_input = key.get_key() if config.log_manual else None
            # 8. シナリオに応じたアクション実行（自動/手動/ボトル回避等）
            if config.log_manual:
                scenario.execute_mode_action(action=action, steer_result=steer_result, bottle=bottle_result, key=key_input)
            else:
                scenario.execute_mode_action(action=action, steer_result=steer_result, bottle=bottle_result)
            # 9. 動画保存・PC送信（必要時のみ）
            if (config.log_save_video or config.log_send_video) and video is not None:
                if not video.process_and_send(frame, steer_result, scenario, action, bottle_result, bottle_masks):
                    print("[ERROR] send_camera_capture failed. Breaking main loop.")
                    break
            # 10. ループ周期調整（MOTOR_SEND_INTERVALサイクルで動作）
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0, MOTOR_SEND_INTERVAL - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            # 追加sleep（念のため）
            total_elapsed = time.time() - loop_start
            if total_elapsed < MOTOR_SEND_INTERVAL:
                time.sleep(MOTOR_SEND_INTERVAL - total_elapsed)
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
    print("Press Ctrl+C to stop")
    config = Config(
        log_sensor=args.record_sensor,
        log_save_video=args.save_video,
        log_send_video=args.send_video,
        log_manual=args.manual
    )
    main(config)

