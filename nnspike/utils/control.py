"""
Line Follower Control Module

This module contains implementations of line-following algorithms using camera-based
methods. These implementations are used for training data collection and real-time 
robot control. By sending input data such as a camera image, the functions return 
the necessary adjustments.

Functions:
    steer_by_camera(frame: np.ndarray, roi: tuple) -> tuple[float, float, float, object]:
        Calculates the steering adjustment based on the difference between the
        center of the Region of Interest (ROI) and the centroid of the largest contour
        in the x-coordinate.

    calculate_theta_from_pixels(offset_pixels: float, image_width: int = 640,
                              sensitivity: float = 0.4) -> float:
        Calculates attitude angle (theta) from pixel offset using simple normalization.

    calculate_adaptive_speed(abs_theta: float, base_power: float = 30,
                          curve_power: float = 20, straight_power: float = 10, curve_threshold: float = 0.0524) -> float:
        Calculates adaptive speed based on curve detection and time-based acceleration.
"""

import math
from collections import deque

# ==== 制御パラメータ（ここだけ編集すればOK）====
THETA_MA_WINDOW = 7  # θ平滑化ウィンドウ長
BASE_POWER = 50
STRAIGHT_POWER = 80
CURVE_POWER = 50
STRAIGHT_THRESHOLD_DEG = 3
CURVE_THRESHOLD_DEG = 10
SENSITIVITY = 0.8
MAX_THETA_DEG = 30
MAX_POWER_DIFF = 20

class ControlCalculator:
    def __init__(self, image_width):
        # ユーザーパラメータを利用
        self.theta_ma_window = THETA_MA_WINDOW
        self.theta_ma_buffer = deque(maxlen=self.theta_ma_window)
        self.base_power = BASE_POWER
        self.straight_power = STRAIGHT_POWER
        self.curve_power = CURVE_POWER
        self.straight_threshold = math.radians(STRAIGHT_THRESHOLD_DEG)
        self.curve_threshold = math.radians(CURVE_THRESHOLD_DEG)
        self.max_theta = math.radians(MAX_THETA_DEG)
        self.image_width = image_width

    def calculate_theta_from_pixels(self, offset_pixels):
        """
        Calculate attitude angle (theta) from pixel offset using normalization.
        SENSITIVITY, image_widthはインスタンス変数・定数から取得。
        """
        image_center_x = self.image_width / 2
        max_offset = image_center_x
        normalized_offset = offset_pixels / max_offset
        theta = normalized_offset * SENSITIVITY
        return theta

    def add_and_get_smoothed_theta(self, theta):
        """
        θ値をバッファに追加し、移動平均で平滑化した値を返す。
        theta_ma_windowでバッファ長を調整可能。
        """
        self.theta_ma_buffer.append(theta)
        if len(self.theta_ma_buffer) > 0:
            return sum(self.theta_ma_buffer) / len(self.theta_ma_buffer)
        else:
            return theta

    def calculate_adaptive_speed(self):
        """
        self.theta_ma_bufferの最新値（平滑化後theta）を使って推奨速度（パワー）を自動調整する。
        """
        if len(self.theta_ma_buffer) == 0:
            abs_theta = 0
        else:
            abs_theta = abs(self.theta_ma_buffer[-1])
        if abs_theta > self.curve_threshold:
            return self.curve_power
        elif abs_theta < self.straight_threshold:
            return self.straight_power
        else:
            return self.base_power

    def calculate_power_adjustment(self, pid_corrected_theta):
        """
        PID補正値（ラジアン）をパワー差分に変換する。
        MAX_POWER_DIFFは定数から取得。
        """
        return int((pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF)
