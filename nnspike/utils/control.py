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
    def __init__(self, roi, image_width, sensitivity, base_power, curve_power, straight_power, curve_threshold, straight_threshold=math.radians(2)):
        self.roi = roi
        self.image_width = image_width
        self.sensitivity = sensitivity
        self.base_power = base_power
        self.curve_power = curve_power
        self.straight_power = straight_power
        self.curve_threshold = curve_threshold
        self.straight_threshold = straight_threshold

    def steer_by_camera(self, frame):
        """
        カメラフレームからROI内の輪郭検出を行い、進行方向の判断に必要な情報を返す。

        引数:
            frame (np.ndarray): カメラから取得したフルフレーム画像（NumPy配列）
            roi (tuple): Region of interest tuple (x1, y1, x2, y2)。

        戻り値:
            tuple[float, float, float, object]: 以下のタプルを返す
                - mx (float): 最大輪郭の重心x座標（ROI内）
                - my (float): 最大輪郭の重心y座標（ROI内）
                - offset_pixels (float): ROI中心から重心までのx方向ピクセルオフセット
                - max_contour (object): 検出された最大輪郭（なければNone）

        主な処理:
            1. ROI（関心領域）をフレームから切り出す
            2. ROIをグレースケール変換
            3. ガウシアンブラーでノイズ除去
            4. 二値化でライン部分を抽出
            5. 収縮・膨張でノイズ除去
            6. 輪郭検出
            7. 最大輪郭の重心(mx, my)を計算
            8. ROI中心からのx方向オフセットを計算
            9. (mx, my, offset_pixels, max_contour)を返す
        """
        # Extract ROI and convert to grayscale
        x1, y1, x2, y2 = self.roi
        roi_area = frame[y1:y2, x1:x2]
        image = cv2.cvtColor(roi_area, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

        # Erode to eliminate noise, Dilate to restore eroded parts of image
        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)

            mu = cv2.moments(max_contour)
            # Add 1e-5 to avoid division by zero
            mx = mu["m10"] / (mu["m00"] + 1e-5)
            my = mu["m01"] / (mu["m00"] + 1e-5)
        else:
            mx = image.shape[1] / 2
            my = image.shape[0] / 2
            max_contour = None    # Calculate offset from center of ROI
        roi_center_x = image.shape[1] / 2
        offset_pixels = mx - roi_center_x

        return mx, my, offset_pixels, max_contour

    def calculate_theta_from_pixels(self, offset_pixels):
        """
        Calculate attitude angle (theta) from pixel offset using simple normalization.
        必ず呼び出し元から渡されたimage_width, sensitivityを使う。
        """
        image_center_x = self.image_width / 2
        max_offset = image_center_x
        normalized_offset = offset_pixels / max_offset
        theta = normalized_offset * self.sensitivity
        return theta

    def calculate_adaptive_speed(self, abs_theta):
        """
        abs_theta（進行方向の絶対角度）のみを使って推奨速度（パワー）を自動調整する。
        ・カーブ判定・直線判定の閾値はインスタンス生成時のパラメータで調整可能。
        ・カラーセンサー値やライン判定は一切参照しない。
        ・カーブ時はcurve_power、直線時はstraight_power、それ以外はbase_powerを返す。
        """
        if abs_theta > self.curve_threshold:
            return self.curve_power
        elif abs_theta < self.straight_threshold:  # 直線判定は任意で調整
            return self.straight_power
        else:
            return self.base_power
