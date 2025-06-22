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

class ControlCalculator:
    def __init__(self, roi, image_width, sensitivity, base_power, curve_power, straight_power, curve_threshold):
        self.roi = roi
        self.image_width = image_width
        self.sensitivity = sensitivity
        self.base_power = base_power
        self.curve_power = curve_power
        self.straight_power = straight_power
        self.curve_threshold = curve_threshold

    def steer_by_camera(self, frame):
        """
        Processes a full camera frame to determine the steering direction based on contour detection within a specified ROI.

        Args:
            frame (np.ndarray): The full camera frame as a NumPy array.
            roi (tuple): Region of interest tuple (x1, y1, x2, y2).

        Returns:
            tuple[float, float, float, object]: A tuple containing:
                - mx (float): The x-coordinate of the centroid of the largest contour.
                - my (float): The y-coordinate of the centroid of the largest contour.
                - offset_pixels (float): Pixel offset from the center of the ROI.
                - max_contour (object): The largest contour detected in the ROI.

        The function performs the following steps:
            1. Extracts the region of interest from the full frame.
            2. Converts the ROI to grayscale.
            3. Applies Gaussian blur to the ROI to reduce noise.
            4. Converts the blurred image to a binary image using thresholding.
            5. Erodes and dilates the binary image to eliminate noise and restore eroded parts.
            6. Finds contours in the processed mask.
            7. Identifies the largest contour based on contour area.
            8. Calculates the moments of the largest contour to find its centroid.
            9. Calculates pixel offset from the center of the ROI.
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

    def calculate_adaptive_speed(self, abs_theta, on_black_line):
        """
        カーブ量(abs_theta)と黒ライン判定に応じて速度を調整する。
        黒ライン上かつカーブでなければstraight_power、カーブ時はcurve_power、それ以外はbase_power。
        """
        if on_black_line and abs_theta <= self.curve_threshold:
            return self.straight_power
        elif abs_theta > self.curve_threshold:
            return self.curve_power
        else:
            return self.base_power
