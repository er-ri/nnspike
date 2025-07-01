"""
ラインフォロワー制御モジュール

このモジュールは、カメラ画像を用いたラインフォロー制御アルゴリズムを実装します。
主に学習データ収集やリアルタイムロボット制御で利用されます。

クラス:
    ControlCalculator:
        画像の中心とラインのオフセットから進行角度（theta）を計算し、
        平滑化や速度調整、パワー差分の計算を行います。
"""

import math
from collections import deque
import json

# ====【現場でよく調整する推奨パラメータ】====
BASE_POWER = 32             # 通常走行時の基準パワー
STRAIGHT_POWER = 30         # 直線判定時のパワー
CURVE_POWER = 23            # カーブ判定時のパワー（20→25で復帰力UP）
STRAIGHT_THRESHOLD_DEG = 3  # 直線とみなす角度しきい値[deg]
CURVE_THRESHOLD_DEG = 10    # カーブとみなす角度しきい値[deg]
SENSITIVITY = 1.0           # θ感度UP（1.0→1.1）

# ====【通常は触らない高度なパラメータ】====
MAX_THETA_DEG = 30          # θの最大値[deg]（パワー補正の正規化用）
MAX_POWER_DIFF = 23         # PID補正による最大パワー差分  # 15→20
MAX_POWER_ADJ_STEP = 5     # パワー差分の1ステップ最大変化量（暴走抑制用）

class ControlCalculator:
    """
    ラインフォロワー用制御計算クラス。
    """

    def __init__(self, image_width, debug=True):
        """
        ControlCalculatorの初期化。

        引数:
            image_width: int 画像幅
            debug: bool デバッグ出力ON/OFF
        """
        # 制御パラメータの初期化（使用順に並べ替え）
        self.image_width = image_width  # 画像幅
        self.base_power = BASE_POWER  # 通常時の基準パワー
        self.straight_power = STRAIGHT_POWER  # 直線時の推奨パワー
        self.curve_power = CURVE_POWER  # カーブ時の推奨パワー
        self.straight_threshold = math.radians(STRAIGHT_THRESHOLD_DEG)  # 直線判定しきい値[rad]
        self.curve_threshold = math.radians(CURVE_THRESHOLD_DEG)  # カーブ判定しきい値[rad]
        self.max_theta = math.radians(MAX_THETA_DEG)  # θ最大値[rad]
        self.debug = debug  # デバッグ出力ON/OFF
        self.last_power_adj = 0  # 前回のパワー差分（暴走抑制用）
        self.prev_power_adj = 0  # 暴走抑制用: 前回のパワー差分

    def calculate_and_smooth_theta(self, offset_pixels):
        """
        オフセットピクセルからθを計算し、そのまま返す（平滑化なし）。
        デバッグ出力も1回でまとめて行う。

        引数:
            offset_pixels: int オフセットピクセル
        戻り値:
            float θ[rad]
        """
        image_center_x = self.image_width / 2
        max_offset = image_center_x
        normalized_offset = offset_pixels / max_offset
        theta = normalized_offset * SENSITIVITY
        if self.debug:
            debug_data = {
                "offset": offset_pixels,
                "theta": round(theta, 4),
                "smoothed": round(theta, 4)
            }
            print(f"[CONTROL_DEBUG] calculate_and_smooth_theta {json.dumps(debug_data, ensure_ascii=False)}")
        return theta

    def calculate_adaptive_speed(self, smoothed_theta):
        # 平滑化後θに応じて推奨速度（パワー）を自動調整
        abs_theta = abs(smoothed_theta)
        if abs_theta > self.curve_threshold:
            speed = self.curve_power  # カーブ時
        elif abs_theta < self.straight_threshold:
            speed = self.straight_power  # 直線時
        else:
            speed = self.base_power  # 通常時
        if self.debug:
            debug_data = {
                "abs_theta": round(abs_theta, 4),
                "speed": speed
            }
            print(f"[CONTROL_DEBUG] calculate_adaptive_speed {json.dumps(debug_data, ensure_ascii=False)}")
        return speed

    def calculate_power_adjustment(self, pid_corrected_theta):
        # PID補正値（ラジアン）をパワー差分に変換（暴走抑制なし）
        power_adj = int((pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF)
        if self.debug:
            debug_data = {
                "pid_theta": round(pid_corrected_theta, 4),
                "power_adj": power_adj
            }
            print(f"[CONTROL_DEBUG] calculate_power_adjustment {json.dumps(debug_data, ensure_ascii=False)}")
        return power_adj
