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

import cv2
import numpy as np
import math

class ControlCalculator:
    def __init__(self, base_power, curve_power, straight_power, curve_threshold, straight_threshold=math.radians(2)):
        self.base_power = base_power
        self.curve_power = curve_power
        self.straight_power = straight_power
        self.curve_threshold = curve_threshold
        self.straight_threshold = straight_threshold

    def calculate_theta_from_pixels(self, offset_pixels, image_width, sensitivity):
        """
        Calculate attitude angle (theta) from pixel offset using normalization.
        image_width, sensitivityは呼び出し元から渡す。
        """
        image_center_x = image_width / 2
        max_offset = image_center_x
        normalized_offset = offset_pixels / max_offset
        theta = normalized_offset * sensitivity
        return theta

    def calculate_adaptive_speed(self, abs_theta):
        """
        abs_theta（進行方向の絶対角度）のみを使って推奨速度（パワー）を自動調整する。
        ・カーブ判定・直線判定の閾値はインスタンス生成時のパラメータで調整可能。
        ・カーブ時はcurve_power、直線時はstraight_power、それ以外はbase_powerを返す。
        """
        if abs_theta > self.curve_threshold:
            return self.curve_power
        elif abs_theta < self.straight_threshold:  # 直線判定は任意で調整
            return self.straight_power
        else:
            return self.base_power
