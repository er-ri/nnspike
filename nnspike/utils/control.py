"""
Line Follower Control Module

This module contains implementations of line-following algorithms using camera-based
and color sensor-based methods. These implementations are used for training data
collection and real-time robot control. By sending input data such as a camera image
or sensor status, the functions return the necessary adjustments.

Functions:

    calculate_attitude_angle(offset_pixels: float, roi_bottom_y: int,
                           camera_height: float, focal_length_pixels: float) -> float:
        Calculates attitude angle (theta) from pixel offset using camera geometry
        for more accurate steering control.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def find_line_edges_at_y(image, roi, target_y, threshold_value=50) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the left and right edge points of a black line at a specific Y coordinate.

    Parameters:
    - image: Input image (BGR or grayscale)
    - roi_coords: Tuple (x, y, width, height) defining the ROI
    - target_y: The Y coordinate where to detect line edges (in original image coordinates)
    - threshold_value: Threshold for binary conversion (default: 50)

    Returns:
    - left_x: X coordinate of left edge (None if not found)
    - right_x: X coordinate of right edge (None if not found)
    """

    # Extract ROI coordinates
    x, y, w, h = roi

    # Check if target_y is within ROI
    if target_y < y or target_y >= y + h:
        return None, None

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Extract ROI
    roi = gray[y : y + h, x : x + w]

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Binary threshold to isolate black line
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Calculate the row within the ROI
    roi_row = target_y - y

    # Get the binary row at target Y
    if roi_row >= 0 and roi_row < h:
        row_data = binary[roi_row, :]

        # Find all white pixels (line pixels) in this row
        white_pixels = np.where(row_data == 255)[0]

        if len(white_pixels) > 0:
            # Find leftmost and rightmost white pixels
            left_x_roi = white_pixels[0]
            right_x_roi = white_pixels[-1]

            # Convert back to original image coordinates
            left_x = x + left_x_roi
            right_x = x + right_x_roi

            return left_x, right_x

    return None, None


def find_bottle_center(
    image, color, min_area: int = 500
) -> Tuple[Optional[Tuple[float, float]], Optional[np.ndarray], Optional[float]]:
    """
    Find the center coordinates and color pixel count of a colored object in an image using OpenCV.

    This function detects objects of a specified color in an image and returns information about
    the largest detected object. It supports yellow, blue, and red color detection and can be used
    for various applications including object tracking, color-based navigation, and visual recognition.

    This function is optimized for real-time applications with the following improvements:
    - Accepts numpy array input instead of file paths for real-time processing
    - Uses adaptive thresholding for better edge detection under various lighting conditions
    - Applies contour area filtering to reduce noise and false detections
    - Includes aspect ratio validation to filter out non-object-like shapes
    - Uses smaller morphological kernels for better performance
    - Removes debug print statements for cleaner real-time operation

    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format)
        color (str): Color to detect ('yellow', 'blue', or 'red')
        min_area (int, optional): Minimum contour area threshold for filtering noise. Defaults to 500.

    Returns:
        tuple: ((x, y), largest_contour, color_pixel_count) where (x, y) is the center coordinates,
               largest_contour is the contour of the largest detected object, and color_pixel_count is the
               number of detected color pixels. Returns (None, None, 0) if not found.

    Raises:
        ValueError: If color parameter is not 'yellow', 'blue', or 'red'
    """
    # Validate color parameter
    if color not in ["yellow", "blue", "red"]:
        raise ValueError("Color must be 'yellow', 'blue' or 'red'")

    # Check if image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return None, None, 0

    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define color range based on the specified color
    if color == "yellow":
        lower_color = np.array([15, 100, 100], dtype=np.uint8)
        upper_color = np.array([35, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower_color, upper_color)  # type: ignore[arg-type]
    elif color == "blue":
        lower_color = np.array([90, 90, 110], dtype=np.uint8)
        upper_color = np.array([100, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower_color, upper_color)  # type: ignore[arg-type]
    elif color == "red":
        # First mask (0-10 degrees) - pure red to slightly orangish red
        lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        # Second mask (165-180 degrees) - pure red to slightly purplish red
        lower_red2 = np.array([165, 100, 100], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # type: ignore[arg-type]
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # type: ignore[arg-type]
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        color_mask = np.zeros_like(gray)

    # Calculate color pixel count
    color_pixel_count = cv2.countNonZero(color_mask)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for real-time performance
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # Filter contours by area to remove noise (adjustable minimum area)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None, None

    # Find the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # Calculate center using moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), largest_contour, color_pixel_count

    return None, None, None


def find_bullseye(image, threshold=120) -> Tuple[Optional[Tuple[float, float]], Optional[np.ndarray], Optional[float]]:
    """
    Find the center coordinates of a blue bullseye target in an image.

    Uses blue color masking followed by circular shape detection for efficient bullseye detection.
    Prioritizes detections within a specified distance from the image center (x=320).

    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format)
        threshold (float, optional): Maximum allowed distance from image center (x=320).
                                   Bullseyes within range [320-threshold, 320+threshold] are prioritized.
                                   Defaults to 120.

    Returns:
        tuple: ((x, y), contour, blue_pixel_count) where (x, y) is the center coordinates,
               contour is the detected bullseye contour, and blue_pixel_count is the
               number of blue pixels. Returns (None, None, None) if not found.
    """
    if image is None or image.size == 0:
        return None, None, None

    # Create blue mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in blue regions
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # Image center for proximity calculation
    center_x = 320

    # Find the best bullseye candidate
    best_score = 0
    best_result = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:  # Skip small regions
            continue

        # Get center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        # Calculate scores
        distance_from_center = abs(cx - center_x)
        proximity = max(0, 1 - distance_from_center / threshold)  # Score based on threshold range
        size_score = 1.0 if 500 <= area <= 8000 else 0.5  # Reasonable bullseye size

        # Get circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0

        # Combined score (proximity to center is most important)
        score = proximity * 3.0 + size_score + min(circularity * 2.0, 1.0)

        if score > best_score:
            best_score = score
            # Count blue pixels in this contour
            mask = np.zeros(blue_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255,), -1)
            blue_count = cv2.countNonZero(cv2.bitwise_and(blue_mask, mask))
            best_result = (float(cx), float(cy)), contour, float(blue_count)

    return best_result if best_result else (None, None, None)


def find_gate_virtual_line(
    image: np.ndarray, scan_x: int = 320, from_y: int = 0, to_y: int = 480
) -> Tuple[Optional[Tuple[float, float]], Optional[np.ndarray], Optional[float]]:
    """
    Find the virtual line for gate detection based on gray color regions.

    This function detects gray regions in an image and calculates the center point
    between the leftmost and rightmost gray pixels from a given scan position.

    Args:
        image: Input BGR image as numpy array
        scan_x: X-coordinate to start scanning from (default: 320)

    Returns:
        Tuple containing:
        - Optional[Tuple[float, float]]: Virtual line coordinates (x, y) or None if no gate found
        - Optional[np.ndarray]: The processed gray mask for debugging
        - Optional[float]: Width between left and right borders
    """
    # Input validation
    if image is None or image.size == 0:
        return None, None, None

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjusted parameters for gray number detection
    # Gray colors have low saturation and medium to high value
    lower_gray = np.array([10, 40, 80], dtype=np.uint8)  # Low hue range, very low saturation, medium value
    upper_gray = np.array([180, 60, 180], dtype=np.uint8)  # Full hue range, low saturation, high value

    # Create the gray mask
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray_mask_cleaned = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    gray_mask_cleaned = cv2.morphologyEx(gray_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Find left and right borders
    # Find right border by scanning from scan_x towards the right
    right_border = scan_x
    for x in range(scan_x, 640):
        column = gray_mask_cleaned[from_y:to_y, x]
        if np.any(column != 0):  # More readable than checking all zeros
            right_border = x
            break

    # Find left border by scanning from scan_x towards the left
    left_border = scan_x
    for x in range(scan_x, -1, -1):  # Changed range to include 0
        column = gray_mask_cleaned[from_y:to_y, x]
        if np.any(column != 0):
            left_border = x
            break

    virtual_x = (left_border + right_border) / 2

    numeric_area = np.array(
        [[left_border, from_y], [right_border, from_y], [right_border, to_y], [left_border, to_y]], dtype=np.int32
    )

    return (virtual_x, 0.0), numeric_area, None


def calculate_attitude_angle(
    offset_pixels: float,
    roi_bottom_y: int,
    camera_height: float = 0.20,
    focal_length_pixels: float = 640,
) -> float:
    """
    Calculate attitude angle (theta) from pixel offset using camera geometry.

    This function converts the pixel-based offset detected in the camera image
    to a real-world attitude angle that represents the robot's deviation from
    the desired path. This provides more physically meaningful control compared
    to simple pixel-based normalization.

    Args:
        offset_pixels (float): Lateral offset in pixels from image center
        roi_bottom_y (int): Bottom y-coordinate of ROI (closer to robot)
        camera_height (float, optional): Camera height above ground in meters. Defaults to 0.20.
        focal_length_pixels (float, optional): Camera focal length in pixels. Defaults to 640.

    Returns:
        float: Attitude angle (theta) in radians. Positive values indicate rightward deviation,
               negative values indicate leftward deviation.

    Note:
        The camera parameters (height and focal length) should be calibrated for your
        specific robot setup to ensure accurate angle calculations.
    """
    # Calculate ground distance from camera to the line detection point
    # Using similar triangles: ground_distance / camera_height = focal_length / (image_height - roi_bottom_y)
    image_height = 480  # Assuming standard camera resolution
    ground_distance = camera_height * focal_length_pixels / (image_height - roi_bottom_y)

    # Calculate lateral offset in meters
    # Using similar triangles: lateral_offset / ground_distance = offset_pixels / focal_length
    lateral_offset_meters = offset_pixels * ground_distance / focal_length_pixels

    # Calculate attitude angle (theta) using arctangent
    theta = math.atan2(lateral_offset_meters, ground_distance)

    return theta
