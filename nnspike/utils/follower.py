"""
Line Follower

This module contains implementations of line-following algorithms using camera-based 
and color sensor-based methods. These implementations are used for training data 
collection for a model. By sending input data such as a camera image or sensor status, 
the functions return the necessary adjustments.

Functions:
    steer_by_camera(roi: np.ndarray) -> tuple[float, dict]:
        Calculates the steering adjustment based on the difference between the 
        center of the Region of Interest (ROI) and the centroid of the largest contour 
        in the x-coordinate.

    steer_by_reflection(reflection: int, threshold: int) -> int:
        Determines the steering adjustment based on light reflectivity detected by 
        a color sensor, guiding the robot to follow the edge of a line.
"""

import cv2
import numpy as np


def steer_by_camera(image: np.ndarray) -> tuple[float, float, dict]:
    """
    Processes a region of interest (ROI) from a camera feed to determine the steering direction based on contour detection.

    Args:
        image (np.ndarray): The region of interest from the camera feed, represented as a NumPy array.

    Returns:
        tuple[float, float, dict]: A tuple containing:
            - mx (float): The x-coordinate of the centroid of the largest contour.
            - my (float): The y-coordinate of the centroid of the largest contour.
            - max_contour (dict): The largest contour detected in the ROI.

    The function performs the following steps:
        1. Applies Gaussian blur to the ROI to reduce noise.
        2. Converts the blurred image to a binary image using thresholding.
        3. Erodes and dilates the binary image to eliminate noise and restore eroded parts.
        4. Finds contours in the processed mask.
        5. Identifies the largest contour based on contour area.
        6. Calculates the moments of the largest contour to find its centroid.
    """
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
        max_contour = None

    return mx, my, max_contour


def steer_by_reflection(reflection: int, threshold: int) -> int:
    """Determines the steering adjustment based on light reflectivity detected by a color sensor.
    The robot follows the edge of a line, turning one way when it detects more light reflectivity
    (whitish color) and the other way when it detects less light reflectivity (darkish color).

    Args:
        reflection (int): Light reflection value (white: 100, black: 0).
        threshold (int): The value to trigger the robot's turning.

    Returns:
        reflection_diff (int): The difference between the detected reflection value and the threshold.
    """
    reflection_diff = reflection - threshold
    return reflection_diff
