"""
ラインフォロワー制御モジュール

このモジュールは、カメラ画像を用いたラインフォロー制御アルゴリズムを実装しています。
主に学習データ収集やリアルタイムロボット制御で利用されます。

クラス:
    ControlCalculator:
        画像の中心とラインのオフセットから進行角度（theta）を計算し、
        平滑化や速度調整、パワー差分の計算を行います。
"""

import math
from collections import deque

# ====【現場でよく調整する推奨パラメータ】====
BASE_POWER = 50             # 通常走行時の基準パワー
STRAIGHT_POWER = 80         # 直線判定時のパワー
CURVE_POWER = 50            # カーブ判定時のパワー
STRAIGHT_THRESHOLD_DEG = 3  # 直線とみなす角度しきい値[deg]
CURVE_THRESHOLD_DEG = 10    # カーブとみなす角度しきい値[deg]
SENSITIVITY = 0.8           # 進行角度θの感度（大きいほど敏感）

# ====【通常は触らない高度なパラメータ】====
THETA_MA_WINDOW = 7         # θ平滑化（移動平均）ウィンドウ長
MAX_THETA_DEG = 30          # θの最大値[deg]（パワー補正の正規化用）
MAX_POWER_DIFF = 20         # PID補正による最大パワー差分

class ControlCalculator:
    def __init__(self, image_width):
        # 制御パラメータの初期化（使用順に並べ替え）
        self.image_width = image_width  # 画像幅
        self.theta_ma_window = THETA_MA_WINDOW  # θ平滑化ウィンドウ長
        self.theta_ma_buffer = deque(maxlen=self.theta_ma_window)  # θ移動平均バッファ
        self.base_power = BASE_POWER  # 通常時の基準パワー
        self.straight_power = STRAIGHT_POWER  # 直線時の推奨パワー
        self.curve_power = CURVE_POWER  # カーブ時の推奨パワー
        self.straight_threshold = math.radians(STRAIGHT_THRESHOLD_DEG)  # 直線判定しきい値[rad]
        self.curve_threshold = math.radians(CURVE_THRESHOLD_DEG)  # カーブ判定しきい値[rad]
        self.max_theta = math.radians(MAX_THETA_DEG)  # θ最大値[rad]

    def calculate_theta_from_pixels(self, offset_pixels):
        # オフセットピクセルから進行角度θ[rad]を計算
        image_center_x = self.image_width / 2
        max_offset = image_center_x
        normalized_offset = offset_pixels / max_offset
        theta = normalized_offset * SENSITIVITY
        return theta

    def add_and_get_smoothed_theta(self, theta):
        # θ値をバッファに追加し、移動平均で平滑化した値を返す
        self.theta_ma_buffer.append(theta)
        if len(self.theta_ma_buffer) > 0:
            return sum(self.theta_ma_buffer) / len(self.theta_ma_buffer)
        else:
            return theta

    def calculate_adaptive_speed(self):
        # 平滑化後θに応じて推奨速度（パワー）を自動調整
        if len(self.theta_ma_buffer) == 0:
            abs_theta = 0
        else:
            abs_theta = abs(self.theta_ma_buffer[-1])
        if abs_theta > self.curve_threshold:
            return self.curve_power  # カーブ時
        elif abs_theta < self.straight_threshold:
            return self.straight_power  # 直線時
        else:
            return self.base_power  # 通常時

    def calculate_power_adjustment(self, pid_corrected_theta):
        # PID補正値（ラジアン）をパワー差分に変換
        return int((pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF)
