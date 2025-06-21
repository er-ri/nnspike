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

    calculate_adaptive_speed(abs_theta: float, current_time: float,
                          straight_line_start_time: float, base_power: float = 30,
                          max_power: float = 50, curve_power: float = 20,
                          curve_threshold: float = 0.0524, acceleration_duration: float = 1.0) -> tuple[float, float]:
        Calculates adaptive speed based on curve detection and time-based acceleration.
"""

import cv2
import numpy as np


def steer_by_camera(
    frame: np.ndarray, roi: tuple
) -> tuple[float, float, float, object]:
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
    x1, y1, x2, y2 = roi
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


def calculate_adaptive_speed(
    abs_theta: float,
    current_time: float,
    straight_line_start_time: float,
    base_power: float = 30,
    max_power: float = 50,
    curve_power: float = 20,
    curve_threshold: float = 0.0524,
    acceleration_duration: float = 1.0
) -> tuple[float, float]:
    """
    Calculate adaptive speed based on curve detection and time-based acceleration.
    
    This function implements dynamic speed control that:
    1. Detects line-off conditions (abs_theta == 0) and uses safe curve power
    2. Detects curves (abs_theta > curve_threshold) and decelerates to curve power
    3. Implements time-based acceleration on straight lines from base power to max power
    
    Args:
        abs_theta (float): Absolute value of current attitude angle in radians
        current_time (float): Current timestamp
        straight_line_start_time (float): Start time of current straight line (None if not on straight line)
        base_power (float, optional): Base power for straight lines. Defaults to 30.
        max_power (float, optional): Maximum power after acceleration. Defaults to 50.
        curve_power (float, optional): Power for curves and line-off conditions. Defaults to 20.
        curve_threshold (float, optional): Threshold to detect curves in radians. Defaults to 0.0524 (~3.0 degrees).
        acceleration_duration (float, optional): Time to accelerate from base to max power. Defaults to 1.0.
    
    Returns:
        tuple[float, float]: A tuple containing:
            - current_power (float): Calculated power value
            - new_straight_line_start_time (float): Updated straight line start time (None if not on straight line)
    """
    
    # Determine speed based on curve detection and time-based acceleration
    if abs_theta == 0:
        # Exception: theta is zero, likely off the line - use CURVE_POWER for safety
        return curve_power, None
    elif abs_theta > curve_threshold:
        # In a curve: rapid deceleration to CURVE_POWER
        return curve_power, None
    else:
        # On straight line: time-based acceleration
        if straight_line_start_time is None:
            # Start of straight line - initialize timer
            return base_power, current_time
        else:
            # Calculate elapsed time on straight line
            elapsed_time = current_time - straight_line_start_time
            
            if elapsed_time >= acceleration_duration:
                # Full acceleration after specified duration
                return max_power, straight_line_start_time
            else:
                # Linear acceleration from base_power to max_power over acceleration_duration
                acceleration_factor = elapsed_time / acceleration_duration
                current_power = base_power + (max_power - base_power) * acceleration_factor
                return current_power, straight_line_start_time

def calculate_theta_from_pixels(
    offset_pixels: float,
    image_width: int = 640,
    sensitivity: float = 0.4
) -> float:
    """
    Calculate attitude angle (theta) from pixel offset using simple normalization.
    
    This function converts the pixel-based offset detected in the camera image
    to an attitude angle using a simple normalization approach. The pixel offset
    is normalized to a range of [-1, 1] and then scaled by a sensitivity factor.
    
    Args:
        offset_pixels (float): Lateral offset in pixels from image center
        image_width (int, optional): Camera image width in pixels. Defaults to 640.
        sensitivity (float, optional): Sensitivity factor for angle conversion. Defaults to 0.4.
    
    Returns:
        float: Attitude angle (theta) in radians. Positive values indicate rightward deviation,
               negative values indicate leftward deviation.
    
    Note:
        The sensitivity factor determines the maximum angle range. With sensitivity=0.4,
        the maximum angle is ±0.4 radians (≈±22.9 degrees).
    """
    # Calculate image center and maximum possible offset
    image_center_x = image_width / 2  # Half of image width
    max_offset = image_center_x  # Maximum possible offset
    
    # Normalize offset_pixels to range [-1, 1] and convert to radians
    normalized_offset = offset_pixels / max_offset  # Range: [-1, 1]
    theta = normalized_offset * sensitivity  # Apply sensitivity scaling
    
    return theta
