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


def get_line_edges_at_y(image, roi, target_y, threshold_value=50) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
    - line_width: Width of the line at this Y position (None if not found)
    """

    # Extract ROI coordinates
    x, y, w, h = roi

    # Check if target_y is within ROI
    if target_y < y or target_y >= y + h:
        return None, None, None

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
            line_width = right_x - left_x + 1

            return left_x, right_x, line_width

    return None, None, None


def get_all_line_edges_at_y(image, roi, target_y, threshold_value=50, max_edges=None):
    """
    Get all detected line edges at a specific Y coordinate.

    Parameters:
    - image: Input image (BGR or grayscale)
    - roi: Tuple (x, y, width, height) defining the ROI
    - target_y: The Y coordinate where to detect line edges (in original image coordinates)
    - threshold_value: Threshold for binary conversion (default: 50)
    - max_edges: Maximum number of edges to return (default: None for all edges)

    Returns:
    - List of x-axis coordinates for detected edges: [int, int, ...]
      Returns empty list if no edges are found.
    """

    # Extract ROI coordinates
    x, y, w, h = roi

    # Check if target_y is within ROI
    if target_y < y or target_y >= y + h:
        return []

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Extract ROI
    roi_img = gray[y : y + h, x : x + w]

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)

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
            # Find continuous segments of white pixels and collect all edges
            edges = []
            segment_start = white_pixels[0]

            for i in range(1, len(white_pixels)):
                # Check if there's a gap between consecutive white pixels
                if white_pixels[i] - white_pixels[i - 1] > 1:
                    # End of current segment
                    segment_end = white_pixels[i - 1]

                    # Convert back to original image coordinates
                    left_edge = x + segment_start
                    right_edge = x + segment_end

                    # Add both left and right edges
                    edges.extend([left_edge, right_edge])

                    # Start new segment
                    segment_start = white_pixels[i]

            # Don't forget the last segment
            segment_end = white_pixels[-1]
            left_edge = x + segment_start
            right_edge = x + segment_end

            # Add both left and right edges
            edges.extend([left_edge, right_edge])

            # Apply limit if specified
            if max_edges is not None and len(edges) > max_edges:
                edges = edges[:max_edges]

            return edges

    return []


def find_bottle_center(image, color, min_area: int = 500) -> Tuple[Optional[Tuple[float, float]], Optional[float], int]:
    """
    Find the center coordinates and color pixel count of a colored object in an image using OpenCV.

    This function detects objects of a specified color in an image and returns information about
    the largest detected object. It supports both yellow and blue color detection and can be used
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
        color (str): Color to detect ('yellow' or 'blue')
        min_area (int, optional): Minimum contour area threshold for filtering noise. Defaults to 500.

    Returns:
        tuple: ((x, y), size, color_pixel_count) where (x, y) is the center coordinates,
               size is the area of the largest contour, and color_pixel_count is the
               number of detected color pixels. Returns (None, None, 0) if not found.

    Raises:
        ValueError: If color parameter is not 'yellow' or 'blue'
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

    # Method 1: Color-based detection (for objects with distinctive colors)
    # Define color range based on the specified color
    if color == "yellow":
        # For yellow objects (same thresholds as detect_color_bottle in camera.py)
        lower_color = np.array([15, 100, 100], dtype=np.uint8)
        upper_color = np.array([35, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
    elif color == "blue":
        # ノートブックで検証した最適な青色抽出範囲
        lower_color = np.array([90, 60, 40])
        upper_color = np.array([140, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
    elif color == "red":
        # backup/20250724/control.pyのfind_bottle_center_with_red_countの閾値を反映
        lower_red1 = np.array([0, 90, 60], dtype=np.uint8)
        upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 90, 60], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        color_mask = np.zeros_like(gray)

    # Calculate color pixel count
    color_pixel_count = cv2.countNonZero(color_mask)

    # Method 2: Edge detection for bottle contours
    # Use adaptive thresholding for better edge detection under various lighting
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)  # Invert to make edges white

    # Combine color and edge information
    combined_mask = cv2.bitwise_or(color_mask, edges)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for real-time performance
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, color_pixel_count

    # Filter contours by area to remove noise (adjustable minimum area)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None, color_pixel_count

    # Find the largest contour (assume it's the bottle)
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # Calculate the size (area) of the largest contour
    contour_size = cv2.contourArea(largest_contour)

    # Additional validation: Check aspect ratio of contour to ensure bottle-like shape
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0

    # Bottles are typically taller than they are wide (aspect ratio > 1)
    if aspect_ratio < 0.8:  # Adjust threshold as needed
        return None, None, color_pixel_count

    # Calculate center using moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size, color_pixel_count

    return None, None, color_pixel_count


def find_bullseye(image, min_area: int = 500, min_circularity: float = 0.7) -> Tuple[Optional[Tuple[float, float]], Optional[float], int]:
    """
    Find the center coordinates, size and pixel count of a bullseye target in an image using OpenCV.

    This function detects bullseye targets (concentric circles) in an image and returns information about
    the detected target. It looks for circular patterns with a blue center circle and outer ring structure,
    which is typical for bullseye targets used in robotics applications.

    The function uses multiple detection methods:
    - Blue color detection for the center circle
    - Circle detection using HoughCircles
    - Contour analysis for shape validation
    - Concentric circle pattern matching

    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format)
        min_area (int, optional): Minimum contour area threshold for filtering noise. Defaults to 500.
        min_circularity (float, optional): Minimum circularity threshold (0-1) for circle validation. Defaults to 0.7.

    Returns:
        tuple: ((x, y), size, pixel_count) where (x, y) is the center coordinates,
               size is the area of the detected bullseye, and pixel_count is the
               number of detected blue pixels in the center. Returns (None, None, 0) if not found.
    """
    # Check if image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return None, None, 0

    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Detect blue center circle with broader range
    # Expand blue color range to be more inclusive
    lower_blue = np.array([80, 30, 30])  # Even more inclusive blue range
    upper_blue = np.array([150, 255, 255])

    # Create mask for blue color
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)  # type: ignore[arg-type]

    # Clean up blue mask
    kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)  # type: ignore[assignment]
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)  # type: ignore[assignment]

    # Calculate blue pixel count
    blue_pixel_count = cv2.countNonZero(blue_mask)

    # Method 2: Circle detection using HoughCircles with multiple parameter sets
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Try multiple HoughCircles parameter sets for better detection
    circle_param_sets = [
        # More sensitive parameters
        {"dp": 1, "minDist": 20, "param1": 20, "param2": 15, "minRadius": 3, "maxRadius": 400},
        # Original parameters
        {"dp": 1, "minDist": 30, "param1": 30, "param2": 20, "minRadius": 5, "maxRadius": 300},
        # Less sensitive parameters for larger features
        {"dp": 2, "minDist": 50, "param1": 50, "param2": 30, "minRadius": 10, "maxRadius": 200},
    ]

    all_circles = []
    for params in circle_param_sets:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=params["dp"],
            minDist=params["minDist"],
            param1=params["param1"],
            param2=params["param2"],
            minRadius=params["minRadius"],
            maxRadius=params["maxRadius"],
        )
        if circles is not None:
            circles_array = np.round(circles[0, :]).astype("int")
            all_circles.extend(circles_array)

    # Remove duplicate circles (circles that are very close to each other)
    unique_circles: list[tuple[int, int, int]] = []
    for circle in all_circles:
        x, y, r = circle
        is_duplicate = False
        for existing in unique_circles:
            ex, ey, er = existing
            distance = math.sqrt((x - ex) ** 2 + (y - ey) ** 2)
            if distance < min(r, er) * 0.5:  # If centers are very close
                is_duplicate = True
                break
        if not is_duplicate:
            unique_circles.append(circle)

    # Method 3: Enhanced contour-based detection with multiple approaches
    thresh_methods = []

    # Adaptive thresholding variants
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh_methods.append(thresh1)

    thresh1_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh_methods.append(thresh1_inv)

    # Otsu's thresholding
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_methods.append(thresh2)

    _, thresh2_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_methods.append(thresh2_inv)

    # Simple binary thresholds at different levels
    for thresh_val in [100, 127, 150, 180]:
        _, thresh_bin = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        thresh_methods.append(thresh_bin)
        _, thresh_bin_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        thresh_methods.append(thresh_bin_inv)

    # Detection scoring and selection with Y-position awareness
    candidates = []
    image_height = image.shape[0]

    def calculate_y_position_score(y_coord, image_height):
        """Calculate Y position score that favors upper regions and penalizes lower regions"""
        y_ratio = y_coord / image_height

        # Bullseye targets are typically in upper 30% of image
        if y_ratio <= 0.3:  # Upper 30% - ideal range
            return 2.0
        elif y_ratio <= 0.5:  # 30-50% - acceptable range
            return 1.5
        elif y_ratio <= 0.7:  # 50-70% - lower acceptable range
            return 1.0
        else:  # Below 70% - heavily penalize
            return 0.1

    def calculate_area_penalty(area, image_area):
        """Penalize excessively large areas that might be false positives"""
        area_ratio = area / image_area
        if area_ratio > 0.15:  # If area is more than 15% of image, heavily penalize
            return 0.2
        elif area_ratio > 0.08:  # If area is more than 8% of image, moderately penalize
            return 0.5
        else:
            return 1.0

    image_area = image_height * image.shape[1]

    # Method A: Blue contours + Circle validation
    if blue_pixel_count > 0:
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for blue_contour in blue_contours:
            blue_area = cv2.contourArea(blue_contour)
            if blue_area < 5:  # Very small threshold
                continue

            # Get center of blue region
            M = cv2.moments(blue_contour)
            if M["m00"] != 0:
                blue_cx = int(M["m10"] / M["m00"])
                blue_cy = int(M["m01"] / M["m00"])

                # Calculate position and area penalties
                y_score = calculate_y_position_score(blue_cy, image_height)
                area_penalty = calculate_area_penalty(blue_area, image_area)

                # Check if this blue center is near any detected circle
                best_circle_match = None
                min_distance = float("inf")

                for x, y, r in unique_circles:
                    distance = math.sqrt((blue_cx - x) ** 2 + (blue_cy - y) ** 2)
                    if distance < r and distance < min_distance:  # Blue center within circle
                        min_distance = distance
                        best_circle_match = (x, y, r)

                if best_circle_match is not None:
                    x, y, r = best_circle_match
                    circle_area = math.pi * r * r
                    # Score based on blue area, circle size, position, and area penalty
                    base_score = blue_area * circle_area / (min_distance + 1)
                    score = base_score * y_score * area_penalty
                    # Use the more accurate blue center instead of circle center for position
                    candidates.append(((blue_cx, blue_cy), circle_area, score, "blue+circle"))
                else:
                    # Use blue center even without circle match
                    base_score = blue_area * 10  # Lower score for blue-only detection
                    score = base_score * y_score * area_penalty
                    candidates.append(((blue_cx, blue_cy), blue_area, score, "blue-only"))

    # Method B: Circle-only detection
    for x, y, r in unique_circles:
        circle_area = math.pi * r * r
        if circle_area >= min_area:
            # Calculate position and area penalties
            y_score = calculate_y_position_score(y, image_height)
            area_penalty = calculate_area_penalty(circle_area, image_area)

            # Check if there are blue pixels near this circle center
            blue_bonus = 0
            if blue_pixel_count > 0:
                center_region_size = max(5, r // 4)
                y1 = max(0, y - center_region_size)
                y2 = min(blue_mask.shape[0], y + center_region_size)
                x1 = max(0, x - center_region_size)
                x2 = min(blue_mask.shape[1], x + center_region_size)
                center_region = blue_mask[y1:y2, x1:x2]
                blue_in_center = cv2.countNonZero(center_region)
                blue_bonus = blue_in_center * 100

            base_score = circle_area + blue_bonus
            score = base_score * y_score * area_penalty
            candidates.append(((x, y), circle_area, score, "circle-only"))

    # Method C: Contour-based detection (relaxed)
    for thresh in thresh_methods:
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area * 0.1:  # Very relaxed area threshold
                continue

            # Very relaxed circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity * 0.2:  # Very lenient
                continue

            # Get contour center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Calculate position and area penalties
                y_score = calculate_y_position_score(cy, image_height)
                area_penalty = calculate_area_penalty(area, image_area)

                # Check for blue pixels in this contour
                blue_bonus = 0
                if blue_pixel_count > 0:
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    overlap = cv2.bitwise_and(blue_mask, mask)
                    blue_in_contour = cv2.countNonZero(overlap)
                    blue_bonus = blue_in_contour * 50

                base_score = area * circularity + blue_bonus
                score = base_score * y_score * area_penalty
                candidates.append(((cx, cy), area, score, "contour"))

    # Select the best candidate
    if not candidates:
        return None, None, blue_pixel_count

    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Return the best candidate
    best_center, best_size, best_score, method = candidates[0]

    return best_center, best_size, blue_pixel_count


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


# --- backup/20250724/control.pyより ---
def find_blue_target_center(
    img,
    blue_hsv_lower=(100, 80, 80),
    blue_hsv_upper=(140, 255, 255),
    gray_hsv_lower=(0, 0, 60),
    gray_hsv_upper=(180, 60, 140),
    blur_kernel=5,
    gray_ellipse_enable=True
):
    """
    青い的（楕円）またはグレー線の中心座標・形状情報を返す統合検出関数（グレーライン補完含む）。
    部分的なグレー線からの楕円推定機能を含む。
    Args:
        img: BGR画像 (numpy.ndarray)
        blue_hsv_lower, blue_hsv_upper: 青色範囲 (HSV)
        gray_hsv_lower, gray_hsv_upper: グレー色範囲 (HSV)
        blur_kernel: 青マスクのメディアンブラーサイズ
    Returns:
        center: (x, y) or None
        area: float or None
        blue_pixel_count: int
    """
    if img is None or img.size == 0:
        return None, None, 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array(blue_hsv_lower), np.array(blue_hsv_upper))
    mask_blue = cv2.medianBlur(mask_blue, blur_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_blue_ellipse = None
    max_blue_area = 0
    best_center = None
    for cnt in contours_blue:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area > 20:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (major, minor), angle = ellipse
                    ratio = major/minor if minor > 0 else 0
                    # 明らかに縦長の形状を除外：縦横比が2.0未満（縦が横の2倍未満）に制限
                    if 0.2 < ratio < 2.0 and major > 5 and minor > 3:
                        if area > max_blue_area:
                            best_blue_ellipse = ellipse
                            max_blue_area = area
                            best_center = (int(cx), int(cy))
                except:
                    continue
    blue_pixel_count = cv2.countNonZero(mask_blue)
    if best_blue_ellipse is not None:
        return best_center, max_blue_area, blue_pixel_count
    if gray_ellipse_enable:
        # グレー楕円もblue_pixel_count=0で返す
        mask_gray = cv2.inRange(hsv, np.array(gray_hsv_lower), np.array(gray_hsv_upper))
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        mask_gray = cv2.dilate(mask_gray, np.ones((5,5), np.uint8), iterations=1)
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipses = []
        for c in contours_gray:
            if len(c) < 10 or c.shape[0] < 5:
                continue
            ellipse_cv = cv2.fitEllipse(c)
            center = (int(np.round(ellipse_cv[0][0])), int(np.round(ellipse_cv[0][1])))
            axes_ = (int(ellipse_cv[1][0]//2), int(ellipse_cv[1][1]//2))
            area = np.pi * axes_[0] * axes_[1]
            if area >= 40000:
                ellipses.append({'center': center, 'area': area})
        if ellipses:
            gray_center = max(ellipses, key=lambda e: e['area'])['center']
            gray_area = max(ellipses, key=lambda e: e['area'])['area']
            return gray_center, gray_area, 0
    return None, None, 0

def get_virtual_line_edges_at_y(img, target_y, line_width=10, image_width=640, fallback_center_x=None, previous_center_x=None, avoidance_preference=None):
    """
    黒色障害物を考慮して仮想ラインの中心座標を計算する関数。
    
    Args:
        img: 入力画像
        target_y: 目標y座標
        line_width: ライン幅（デフォルト10）
        image_width: 画像幅（デフォルト640）
        fallback_center_x: フォールバック中心x座標
        previous_center_x: 前回の中心x座標
        avoidance_preference: 単一障害物の回避方向の優先設定
                            'left': 左回避を優先
                            'right': 右回避を優先
                            None: 画像中心基準で自動決定（デフォルト）
    
    Returns:
        int: 計算された仮想ライン中心のx座標
    """
    # --- 進路決定パラメータの初期化 ---
    if fallback_center_x is None:
        fallback_center_x = image_width // 2  # 画像中央をデフォルト中心
    if previous_center_x is not None:
        # 前回中心があれば有効範囲を狭める
        valid_center_min = max(50, previous_center_x - 50)
        valid_center_max = min(image_width - 50, previous_center_x + 50)
    else:
        valid_center_min = 50
        valid_center_max = image_width - 50

    # --- 画像の前処理（2値化・ノイズ除去）---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール変換
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # 適応的2値化
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bin_cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_noise)  # 小ノイズ除去
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    bin_dilated = cv2.dilate(bin_cleaned, kernel_dilate, iterations=1)  # 領域拡張
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    bin_final = cv2.morphologyEx(bin_dilated, cv2.MORPH_CLOSE, kernel_close)  # 領域の穴埋め

    # --- 輪郭抽出と領域情報リスト化 ---
    contours, _ = cv2.findContours(bin_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # 小さい領域や特定範囲を除外し、認識領域のみ抽出
        if (50 < area < 20000 and w >= 15 and h >= 15 and y <= 450 and not (y >= 330 and 120 <= x <= 520)):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [cnt], 255)
            region_pixels = gray[mask == 255]
            # 暗さ判定を改善：平均値だけでなく最小値も考慮
            avg_brightness = np.mean(region_pixels) if len(region_pixels) > 0 else 255
            min_brightness = np.min(region_pixels) if len(region_pixels) > 0 else 255
            avg_darkness = 255 - avg_brightness
            # 黒色障害物の検出を強化：最小値が非常に暗い場合はボーナス
            darkness_bonus = 80 if min_brightness < 50 else 0  # ボーナスを50→80に増加
            # さらに、平均輝度が低い場合も追加ボーナス
            if avg_brightness < 100:
                darkness_bonus += 50  # 平均輝度が低い場合の追加ボーナス
            effective_darkness = avg_darkness + darkness_bonus
            # --- 極端に横長かつ暗い領域を除外 ---
            if w / h > 10 and effective_darkness > 100:
                continue
            detected_regions.append({
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
                'center': (x + w//2, y + h//2),
                'darkness': effective_darkness,
                'priority': effective_darkness * (area / 1000)
            })

    # --- 小領域のマージ処理 ---
    merged = []
    processed = set()
    x_merge_threshold = 80  # x座標が80px未満の差なら同一障害物とみなす
    for i, region in enumerate(detected_regions):
        if i in processed:
            continue
        current_group = [region]
        processed.add(i)
        for j, other in enumerate(detected_regions):
            if j in processed or j <= i:
                continue
            dx = abs(region['center'][0] - other['center'][0])
            dy = region['center'][1] - other['center'][1]
            distance = (dx*dx + dy*dy) ** 0.5
            # x座標が近いものは同一障害物としてまとめる
            if dx < x_merge_threshold:
                current_group.append(other)
                processed.add(j)
            # 近接かつ小面積同士もまとめる（従来ロジック）
            elif distance < 100 and (region['area'] < 1000 or other['area'] < 1000):
                current_group.append(other)
                processed.add(j)
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            min_x = min(r['x'] for r in current_group)
            min_y = min(r['y'] for r in current_group)
            max_x = max(r['x'] + r['w'] for r in current_group)
            max_y = max(r['y'] + r['h'] for r in current_group)
            max_darkness = max(r['darkness'] for r in current_group)
            total_area = sum(r['area'] for r in current_group)
            merged.append({
                'x': min_x, 'y': min_y, 'w': max_x - min_x, 'h': max_y - min_y,
                'area': total_area,
                'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
                'darkness': max_darkness,
                'priority': max_darkness * (total_area / 1000)
            })
    detected_regions = merged

    # --- 進路中心計算ロジック（黒色障害物回避優先） ---
    trajectory_center_x = None
    min_safe_gap = 100  # 最小安全幅（80→100に拡大：黒色障害物対策）
    if not detected_regions:
        # 領域がなければ前回値または中央
        trajectory_center_x = previous_center_x if previous_center_x is not None else fallback_center_x
    else:
        regions_sorted = sorted(detected_regions, key=lambda x: x['x'])  # x座標順
        back_regions = [r for r in detected_regions if r['y'] <= target_y - 50]  # 奥側領域
        high_priority_threshold = 80  # 黒色障害物検出をさらに強化（100→80に下げる）
        
        # 黒色障害物を最優先で検出・回避（超保守的戦略）
        dark_obstacles = [r for r in detected_regions if r['darkness'] > high_priority_threshold]
        if dark_obstacles:
            # 黒色障害物がある場合は、それらを最優先で回避
            dark_sorted = sorted(dark_obstacles, key=lambda x: x['x'])  # x座標順
            if len(dark_sorted) >= 2:
                # 黒色障害物が複数ある場合
                best_gap = None
                for i in range(len(dark_sorted) - 1):
                    left_region = dark_sorted[i]
                    right_region = dark_sorted[i + 1]
                    left_edge = left_region['x'] + left_region['w']
                    right_edge = right_region['x']
                    safety_margin = 100  # 基本マージン（80→100に拡大）
                    gap_width = right_edge - left_edge - safety_margin
                    if gap_width >= min_safe_gap:
                        candidate_x = (left_edge + right_edge) // 2
                        # 有効範囲を適度に縮小して過度な回避を防ぐ
                        expanded_min = max(20, previous_center_x - 80) if previous_center_x else 20
                        expanded_max = min(image_width - 20, previous_center_x + 80) if previous_center_x else image_width - 20
                        if expanded_min <= candidate_x <= expanded_max:
                            if best_gap is None or gap_width > best_gap['width']:
                                best_gap = {'width': gap_width, 'center': candidate_x}
                if best_gap:
                    trajectory_center_x = best_gap['center']
            else:
                # 単一の黒色障害物の場合（極めて保守的な回避）
                region = dark_sorted[0]
                contour_center = region['center'][0]
                image_center = image_width // 2
                safe_distance_from_obstacle = 140  # 黒色障害物からの安全距離（140ピクセル）
                # 前回中心から±60ピクセルの範囲で回避先を制限（過度な移動を防ぐ）
                expanded_min = max(20, previous_center_x - 60) if previous_center_x else 20
                expanded_max = min(image_width - 20, previous_center_x + 60) if previous_center_x else image_width - 20
                
                # 回避方向の決定（安全性を優先）
                if avoidance_preference == 'left':
                    # 左回避を優先：但し障害物が極端に左にある場合は右回避も検討
                    left_candidate = contour_center - safe_distance_from_obstacle
                    right_candidate = contour_center + safe_distance_from_obstacle
                    
                    # 障害物が極端に左（x <= 120）にある場合は右回避を検討
                    if contour_center <= 120:
                        # 右回避の方が安全かチェック
                        if right_candidate <= expanded_max:
                            candidate_x = right_candidate  # 右回避を選択
                        else:
                            candidate_x = left_candidate   # 左回避のまま
                    else:
                        candidate_x = left_candidate
                        
                elif avoidance_preference == 'right':
                    # 右回避を優先：但し障害物が極端に右にある場合は左回避も検討
                    left_candidate = contour_center - safe_distance_from_obstacle
                    right_candidate = contour_center + safe_distance_from_obstacle
                    
                    # 障害物が極端に右（x >= 520）にある場合は左回避を検討
                    if contour_center >= 520:
                        # 左回避の方が安全かチェック
                        if left_candidate >= expanded_min:
                            candidate_x = left_candidate   # 左回避を選択
                        else:
                            candidate_x = right_candidate  # 右回避のまま
                    else:
                        candidate_x = right_candidate
                        
                else:
                    # 従来の自動判定：画像中心を基準に決定
                    if contour_center < image_center:
                        candidate_x = contour_center + safe_distance_from_obstacle
                    else:
                        candidate_x = contour_center - safe_distance_from_obstacle
                
                # まず拡大範囲で試す
                if expanded_min <= candidate_x <= expanded_max:
                    trajectory_center_x = candidate_x
                # 拡大範囲でもダメなら端に寄せる
                elif candidate_x < expanded_min:
                    trajectory_center_x = expanded_min
                elif candidate_x > expanded_max:
                    trajectory_center_x = expanded_max
        
        # 黒色障害物による回避が決まらなかった場合のみ、従来ロジックを実行
        if trajectory_center_x is None and len(back_regions) >= 2:
            # 奥側2領域から中心に近いペアを選び、左右は順序通り（pair[0]=left, pair[1]=right）
            img_center_x = image_width // 2
            regions_by_center = sorted(back_regions, key=lambda r: abs(r['center'][0] - img_center_x))
            pair = regions_by_center[:2]
            left = pair[0]
            right = pair[1]
            left_edge = left['x'] + left['w']
            right_edge = right['x']
            safety_margin = 80  # 基本安全マージン拡大（60→80：黒色障害物対策）
            if left['darkness'] > high_priority_threshold:
                safety_margin += 60  # 暗い障害物：追加で60（合計140）
            if right['darkness'] > high_priority_threshold:
                safety_margin += 60  # 暗い障害物：追加で60（合計140）
            gap_width = right_edge - left_edge - safety_margin
            # 安全幅を満たす場合のみ中心候補
            if gap_width >= min_safe_gap:
                candidate_x = (left_edge + right_edge) // 2
                if valid_center_min <= candidate_x <= valid_center_max:
                    trajectory_center_x = candidate_x
        # 他のペア探索（全領域）
        if trajectory_center_x is None and len(regions_sorted) >= 2:
            best_gap = None
            for i in range(len(regions_sorted) - 1):
                left_region = regions_sorted[i]
                right_region = regions_sorted[i + 1]
                left_edge = left_region['x'] + left_region['w']
                right_edge = right_region['x']
                safety_margin = 80  # 基本安全マージン拡大（60→80：黒色障害物対策）
                if left_region['darkness'] > high_priority_threshold:
                    safety_margin += 60  # 暗い障害物：追加で60（合計140）
                if right_region['darkness'] > high_priority_threshold:
                    safety_margin += 60  # 暗い障害物：追加で60（合計140）
                gap_width = right_edge - left_edge - safety_margin
                if gap_width >= min_safe_gap:
                    candidate_x = (left_edge + right_edge) // 2
                    if valid_center_min <= candidate_x <= valid_center_max:
                        priority_bonus = (left_region['darkness'] + right_region['darkness']) / 10
                        effective_width = gap_width + priority_bonus
                        if best_gap is None or effective_width > best_gap['width']:
                            best_gap = {'width': effective_width, 'center': candidate_x}
            if best_gap:
                trajectory_center_x = best_gap['center']
        # 単一領域時の中心決定（黒色障害物対策強化）
        if trajectory_center_x is None and len(detected_regions) == 1:
            region = detected_regions[0]
            contour_center = region['center'][0]
            image_center = image_width // 2
            # 暗い障害物だけ少し大きく、それ以外は100px
            safe_distance_from_obstacle = 140 if region['darkness'] > high_priority_threshold else 100
            # 前回中心から±60ピクセルの範囲で回避先を制限（過度な移動を防ぐ）
            expanded_min = max(20, previous_center_x - 60) if previous_center_x else 20
            expanded_max = min(image_width - 20, previous_center_x + 60) if previous_center_x else image_width - 20
            
            # 回避方向の決定（安全性を優先）
            if avoidance_preference == 'left':
                # 左回避を優先：但し障害物が極端に左にある場合は右回避も検討
                left_candidate = contour_center - safe_distance_from_obstacle
                right_candidate = contour_center + safe_distance_from_obstacle
                
                # 障害物が極端に左（x <= 120）にある場合は右回避を検討
                if contour_center <= 120:
                    # 右回避の方が安全かチェック
                    if right_candidate <= expanded_max:
                        candidate_x = right_candidate  # 右回避を選択
                    else:
                        candidate_x = left_candidate   # 左回避のまま
                else:
                    candidate_x = left_candidate
                    
            elif avoidance_preference == 'right':
                # 右回避を優先：但し障害物が極端に右にある場合は左回避も検討
                left_candidate = contour_center - safe_distance_from_obstacle
                right_candidate = contour_center + safe_distance_from_obstacle
                
                # 障害物が極端に右（x >= 520）にある場合は左回避を検討
                if contour_center >= 520:
                    # 左回避の方が安全かチェック
                    if left_candidate >= expanded_min:
                        candidate_x = left_candidate   # 左回避を選択
                    else:
                        candidate_x = right_candidate  # 右回避のまま
                else:
                    candidate_x = right_candidate
                    
            else:
                # 従来の自動判定：画像中心を基準に決定
                if contour_center < image_center:
                    candidate_x = contour_center + safe_distance_from_obstacle
                else:
                    candidate_x = contour_center - safe_distance_from_obstacle
            
            # まず拡大範囲で試す
            if expanded_min <= candidate_x <= expanded_max:
                trajectory_center_x = candidate_x
            # 拡大範囲でもダメなら端に寄せる
            elif candidate_x < expanded_min:
                trajectory_center_x = expanded_min
            elif candidate_x > expanded_max:
                trajectory_center_x = expanded_max
        # どれにも該当しない場合は前回値または中央
        if trajectory_center_x is None:
            trajectory_center_x = previous_center_x if previous_center_x is not None else fallback_center_x

    # --- 最終的な中心値を画像範囲内にクリップ ---
    trajectory_center_x = max(line_width//2, min(image_width - line_width//2 - 1, trajectory_center_x))
    return trajectory_center_x

def get_is_blue_line_at_y(img, target_y, min_run=30):
    """
    指定したy座標（target_y）で、HSV条件に合致する青ピクセルがmin_run個以上連続していればTrue、そうでなければFalseを返す。
    画像全体のx方向を横断して判定する。ノイズ除去や細いラインの検出に有効。

    Args:
        img (np.ndarray): BGR画像
        target_y (int): 判定するy座標（画像全体基準）
        min_run (int, optional): 青ピクセルの最小連続数（デフォルト30）。

    Returns:
        bool: min_run個以上連続した青ピクセルがあればTrue、なければFalse
    """
    if not (0 <= target_y < img.shape[0]):
        return False
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_hsv_lower = (100, 80, 80)
    blue_hsv_upper = (140, 255, 255)
    line_hsv = img_hsv[target_y, :]
    blue_mask = np.all([(blue_hsv_lower[i] <= line_hsv[:,i]) & (line_hsv[:,i] <= blue_hsv_upper[i]) for i in range(3)], axis=0)
    # 連続する青ピクセル数がmin_run以上あるか判定
    max_run = 0
    current_run = 0
    for v in blue_mask:
        if v:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run >= min_run

# x=320の中心ラインが青的（青い楕円）にヒットしたらTrueを返す関数
def is_x320_on_blue_target(img, x_tolerance=40):
    """
    画像内の青的（楕円）の中心がx=320±x_toleranceの範囲にあればTrueを返す。
    青的が見つからなければFalse。
    Args:
        img: BGR画像 (numpy.ndarray)
        x_tolerance: 許容するx方向の誤差幅（ピクセル）
    Returns:
        bool: x=320付近に青的があればTrue、なければFalse
    """
    if img is None or img.size == 0:
        return False
    blue_hsv_lower = (100, 80, 80)
    blue_hsv_upper = (140, 255, 255)
    blur_kernel = 5
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array(blue_hsv_lower), np.array(blue_hsv_upper))
    mask_blue = cv2.medianBlur(mask_blue, blur_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = None
    max_blue_area = 0
    for cnt in contours_blue:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area > 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (major, minor), angle = ellipse
                    ratio = major/minor if minor > 0 else 0
                    if 0.2 < ratio < 5.0 and major > 5 and minor > 3:
                        if area > max_blue_area:
                            max_blue_area = area
                            best_center = (int(cx), int(cy))
                except:
                    continue
    if best_center is None:
        return False
    cx, cy = best_center
    if abs(cx - 320) <= x_tolerance:
        return True
    return False

# x=320の中心ラインが赤的（赤い楕円）にヒットしたらTrueを返す関数
def is_x320_on_red_target(img, x_tolerance=40):
    """
    画像内の赤的（楕円）の中心がx=320±x_toleranceの範囲にあればTrueを返す。
    赤的が見つからなければFalse。
    Args:
        img: BGR画像 (numpy.ndarray)
        x_tolerance: 許容するx方向の誤差幅（ピクセル）
    Returns:
        bool: x=320付近に赤的があればTrue、なければFalse
    """
    if img is None or img.size == 0:
        return False
    # 赤色のHSV範囲（2区間）
    red_hsv_lower1 = (0, 90, 60)
    red_hsv_upper1 = (15, 255, 210)
    red_hsv_lower2 = (175, 90, 60)
    red_hsv_upper2 = (180, 255, 210)
    blur_kernel = 5
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(red_hsv_lower1), np.array(red_hsv_upper1))
    mask2 = cv2.inRange(hsv, np.array(red_hsv_lower2), np.array(red_hsv_upper2))
    mask_red = cv2.bitwise_or(mask1, mask2)
    mask_red = cv2.medianBlur(mask_red, blur_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = None
    max_red_area = 0
    for cnt in contours_red:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area > 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (major, minor), angle = ellipse
                    ratio = major/minor if minor > 0 else 0
                    if 0.2 < ratio < 5.0 and major > 5 and minor > 3:
                        if area > max_red_area:
                            max_red_area = area
                            best_center = (int(cx), int(cy))
                except:
                    continue
    if best_center is None:
        return False
    cx, cy = best_center
    if abs(cx - 320) <= x_tolerance:
        return True
    return False

def get_red_target_center_x(img):
    """
    画像内の赤的（楕円）の中心x座標を返す。
    赤的が見つからなければNoneを返す。
    Args:
        img: BGR画像 (numpy.ndarray)
    Returns:
        int or None: 赤的の中心x座標、見つからなければNone
    """
    if img is None or img.size == 0:
        return None
    # 赤色のHSV範囲（2区間）
    red_hsv_lower1 = (0, 90, 60)
    red_hsv_upper1 = (15, 255, 210)
    red_hsv_lower2 = (175, 90, 60)
    red_hsv_upper2 = (180, 255, 210)
    blur_kernel = 5
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(red_hsv_lower1), np.array(red_hsv_upper1))
    mask2 = cv2.inRange(hsv, np.array(red_hsv_lower2), np.array(red_hsv_upper2))
    mask_red = cv2.bitwise_or(mask1, mask2)
    mask_red = cv2.medianBlur(mask_red, blur_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center_x = None
    max_red_area = 0
    for cnt in contours_red:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area > 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (major, minor), angle = ellipse
                    ratio = major/minor if minor > 0 else 0
                    # is_x320_on_red_targetと同じ条件
                    if 0.2 < ratio < 5.0 and major > 5 and minor > 3:
                        if area > max_red_area:
                            max_red_area = area
                            best_center_x = int(cx)
                except Exception:
                    continue
    return best_center_x

# 黒ラインの長さや位置で判定する関数（画像直接渡し、条件はプライベート変数）
def is_left_black_line_detected(img, course):
    _min_width = 60
    _min_height = 150
    _min_aspect = 2
    _min_area = 8000
    _roi = (0, 80, 140, 420)
    #roi_left = (500, 80, 640, 420)  # 画像幅640前提
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    # courseがleftの時はimgを左右反転
    if course == 'left':
        img = cv2.flip(img, 1)
    # --- 元のグレースケール処理 ---
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # --- CLAHEコントラスト強調方式（get_virtual_line_target_xと同じ） ---
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img_clahe = clahe.apply(img_gray)
    # _, mask = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # h, w = mask.shape
    # mask[:, w//2:] = 0
    # --- 強めノイズ除去（元の処理） ---
    # mask = cv2.medianBlur(mask, 9)
    # mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=3)
    # --- 弱め＋穴埋め（小さいカーネル・回数少なめ＋クロージング） ---
    # mask = cv2.medianBlur(mask, 5)
    # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # --- get_line_edges_at_y方式（control_preprocess_image利用）---
    mask = control_preprocess_image(
        img,
        grayscale=True,
        blur_type="gaussian",
        blur_ksize=5,
        threshold=80,
        threshold_type="binary_inv",
        noise_removal="none"
    )
    h, w = mask.shape
    x0, y0, x1, y1 = _roi
    mask_roi = np.zeros_like(mask)
    mask_roi[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = hh / (ww + 1e-5)
        # ROI内で幅・高さ・アスペクト比・面積のみで判定
        if ww >= _min_width and hh >= _min_height and aspect >= _min_aspect and area >= _min_area:
            return True
    return False

def is_general_horizontal_line_detected(img):
    """
    x=320と交差する一般的な水平黒ラインが検出されたらTrueを返す関数
    ROI: y0から540まで全体、x=320との交差必須、90度に近い角度を重視
    
    Args:
        img: BGR画像 (numpy.ndarray)
    
    Returns:
        bool: x=320と交差し90度に近い水平黒ラインが検出されればTrue、なければFalse
    """
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    
    # 一般的な水平ライン検出パラメータ
    _min_width = 150      # ノートブック準拠: 幅条件
    _min_height = 10      # ノートブック準拠: 高さ条件
    _max_aspect = 0.2     # ノートブック準拠: アスペクト比（高さ/幅）
    _min_area = 3000      # ノートブック準拠: 面積条件
    _angle_threshold = 10  # 0度±10または90度±10を許容
    _center_x = 320
    _roi = (200, 0, 440, 540)  # ノートブック準拠ROI
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OTSU自動閾値
    _, black_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去（軽め）
    black_mask = cv2.medianBlur(black_mask, 5)
    black_mask = cv2.dilate(black_mask, np.ones((5, 5), np.uint8), iterations=1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # ROI適用
    x0, y0, x1, y1 = _roi
    mask_roi = np.zeros_like(black_mask)
    mask_roi[y0:y1, x0:x1] = black_mask[y0:y1, x0:x1]
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = height / width if width > 0 else float('inf')
        # 角度計算
        angle = None
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            angle_raw = rect[2]
            # minAreaRectの仕様: -90〜0度、短辺がx軸方向に近い場合-90、長辺がx軸方向に近い場合0
            if angle_raw < -45:
                angle_norm = 90 + angle_raw  # 水平に近い場合0付近、垂直に近い場合90付近
            else:
                angle_norm = angle_raw  # 0付近
            angle_from_0 = abs(angle_norm)
            angle_from_90 = abs(abs(angle_norm) - 90)
        else:
            angle_from_0 = 0
            angle_from_90 = 90
        # x=320との交差判定（y座標制限なし）
        crosses_center = (x <= _center_x <= x + width)
        # 0度±10または90度±10を許容
        angle_ok = (angle_from_0 <= _angle_threshold) or (angle_from_90 <= _angle_threshold)
        if (
            width >= _min_width and 
            height >= _min_height and 
            aspect_ratio <= _max_aspect and 
            area >= _min_area and 
            crosses_center and
            angle_ok
        ):
            return True
    return False

def is_horizontal_black_line_detected(img, intersection_y=450):
    """
    x=320を通り、指定されたy座標と交差する水平黒ラインが検出されたらTrueを返す関数
    frame_1909を未検出、frame_1910を検出するようにバランス調整された実装
    
    Args:
        img: BGR画像 (numpy.ndarray)
        intersection_y: 交差判定するy座標 (int, default=450)
    
    Returns:
        bool: x=320を通り、指定されたy座標と交差する水平黒ラインが検出されればTrue、なければFalse
    """
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    
    # バランス調整されたパラメータ
    _min_width = 400      
    _min_height = 50     # ★75以下に下げて「h=75」もTrueになるよう調整
    _max_aspect = 0.4     
    _min_area = 23000     # frame_1909(22684)と1910(23996)の間に設定
    _center_x = 320
    _roi = (100, 300, 540, 540)
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OTSU自動閾値
    _, black_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去
    black_mask = cv2.medianBlur(black_mask, 9)
    black_mask = cv2.dilate(black_mask, np.ones((7, 7), np.uint8), iterations=2)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    # ROI適用
    x0, y0, x1, y1 = _roi
    mask_roi = np.zeros_like(black_mask)
    mask_roi[y0:y1, x0:x1] = black_mask[y0:y1, x0:x1]

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = height / width if width > 0 else float('inf')
        crosses_center = (x <= _center_x <= x + width)
        crosses_intersection_y = (y <= intersection_y <= y + height)

        # y_crossがTrueなら無条件で検出
        if crosses_intersection_y:
            return True
        else:
            # 通常の厳しい条件
            if (width >= _min_width and height >= _min_height and aspect_ratio <= _max_aspect and area >= _min_area and crosses_center):
                return True
    return False

def is_vertical_black_line_detected(img):
    """
    x=320と交差する一般的な水平黒ラインが検出されたらTrueを返す関数
    ROI: y0から540まで全体、x=320との交差必須、90度に近い角度を重視
    
    Args:
        img: BGR画像 (numpy.ndarray)
    
    Returns:
        bool: x=320と交差し90度に近い水平黒ラインが検出されればTrue、なければFalse
    """
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    
    # 一般的な垂直ライン検出パラメータ（ノートブックと同期）
    _min_width = 70      # 幅条件（70px以上に緩和）
    _min_height = 200    # 高さ条件（200px以上に緩和）
    _min_aspect = 1.8    # アスペクト比（1.8以上に緩和）
    _min_area = 14000     # 面積条件（14000px^2以上に緩和）
    _center_x = 320
    _center_tolerance = 60  # x=320±60px
    _roi = (200, 200, 440, 540)  # 画像下側ROI
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OTSU自動閾値
    _, black_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去（軽め）
    black_mask = cv2.medianBlur(black_mask, 5)
    black_mask = cv2.dilate(black_mask, np.ones((5, 5), np.uint8), iterations=1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # ROI適用
    x0, y0, x1, y1 = _roi
    mask_roi = np.zeros_like(black_mask)
    mask_roi[y0:y1, x0:x1] = black_mask[y0:y1, x0:x1]
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = height / width if width > 0 else float('inf')
        # x=320±_center_toleranceを通るか
        line_center_x = x + width // 2
        crosses_center = abs(line_center_x - _center_x) <= _center_tolerance

        # デバッグ出力

        if (
            width >= _min_width and 
            height >= _min_height and 
            aspect_ratio >= _min_aspect and 
            area >= _min_area and 
            crosses_center
        ):
            return True
    return False

def get_virtual_line_target_x(img, previous_center_x=None):
    # ROI座標（仮想ライン検出範囲）
    x1, y1, x2, y2 = 100, 150, 540, 330
    roi_w, roi_h = x2 - x1, y2 - y1
    # グレースケール＋CLAHE（コントラスト強調）
    img_gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_full = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe_full = clahe_full.apply(img_gray_full)
    # OTSUで2値化
    _, mask_full = cv2.threshold(img_clahe_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ノイズ除去＋領域強調
    mask_blur = cv2.medianBlur(mask_full, 5)
    mask_dilate = cv2.dilate(mask_blur, np.ones((5, 5), np.uint8), iterations=1)
    mask_close = cv2.morphologyEx(mask_dilate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # ROI抽出
    mask = mask_close[y1:y2, x1:x2]
    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    max_aspect = 5.0
    filtered_contours = []
    rect_centers_x = []
    rect_centers_y = []
    rects = []
    # 面積・アスペクト比・最大面積でフィルタ
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h if h > 0 else 0
        if area < min_area:
            continue
        if aspect > max_aspect:
            continue
        if area >= 8000:
            continue
        cx = x + w // 2
        cy = y + h // 2
        filtered_contours.append(cnt)
        rect_centers_x.append(cx)
        rect_centers_y.append(cy)
        rects.append((x, y, w, h))
    # x座標でグループ化
    x_merge_threshold = 150
    merged_groups = []
    used = set()
    for i, cx in enumerate(rect_centers_x):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j, cx2 in enumerate(rect_centers_x):
            if j in used or j == i:
                continue
            if abs(cx - cx2) < x_merge_threshold:
                group.append(j)
                used.add(j)
        merged_groups.append(group)
    # グループごとに外接矩形と面積計算
    merged_rects = []
    merged_areas = []
    for group in merged_groups:
        xs, ys, ws, hs = [], [], [], []
        total_area = 0
        for idx in group:
            x, y, w, h = cv2.boundingRect(filtered_contours[idx])
            xs.append(x)
            ys.append(y)
            ws.append(x+w)
            hs.append(y+h)
            total_area += w * h
        if xs:
            min_x = min(xs)
            min_y = min(ys)
            max_x = max(ws)
            max_y = max(hs)
            merged_rects.append((min_x, min_y, max_x-min_x, max_y-min_y))
            merged_areas.append(total_area)
    # 最大面積グループの端点からtarget_x算出
    if merged_rects:
        max_idx = np.argmax(merged_areas)
        rect = merged_rects[max_idx]
        min_x, min_y, w, h = rect
        max_x = min_x + w
        left_edge_x = x1 + min_x
        right_edge_x = x1 + max_x
        group_center_x = x1 + min_x + w // 2
        group_center_y = y1 + min_y + h // 2
        if group_center_x < 320:
            edge_x = right_edge_x
            target_x = edge_x + 150
        else:
            edge_x = left_edge_x
            target_x = edge_x - 150
    else:
        target_x = 320  # 障害物なし時は中央
    
    # previous_center_xによる極端なジャンプ制限
    if previous_center_x is not None:
        max_delta = 30  # 許容する最大変化量
        if abs(target_x - previous_center_x) > max_delta:
            if target_x > previous_center_x:
                target_x = previous_center_x + max_delta
            else:
                target_x = previous_center_x - max_delta
    return target_x

def control_preprocess_image(
    image,
    roi=None,
    grayscale=True,
    colorspace=None, # None, 'HSV', 'GRAY' など
    blur_type="gaussian",
    blur_ksize=5,
    threshold=80,
    threshold_type="binary_inv",
    mask=None,
    noise_removal="none",
    clahe=False
):
    """
    画像前処理（get_line_edges_at_yと完全同一仕様）
    - グレースケール化（grayscale=True）
    - GaussianBlur（blur_type='gaussian', blur_ksize=5）
    - 二値化（threshold_type='binary_inv', threshold=80）
    - ノイズ除去はデフォルトでなし（noise_removal='none'）
    - ROI抽出（roi指定時のみ）
    - その他用途でパラメータ調整可
    """
    img = image.copy()
    if colorspace == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe_obj.apply(img)
    if blur_type == "gaussian":
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    elif blur_type == "median":
        img = cv2.medianBlur(img, blur_ksize)
    if threshold is not None:
        if threshold_type == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif threshold_type == "binary":
            _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    if mask is not None:
        img = cv2.bitwise_and(img, mask)
    if roi is not None:
        x, y, w, h = roi
        img = img[y : y + h, x : x + w]
    if noise_removal == "dilate":
        img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
    elif noise_removal == "close":
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return img

def get_color_mask(image, color, pattern=None):
    """
    指定色のHSVマスクを返す（yellow, blue, red対応）。
    patternはbottle/line/targetのみ。未指定時はbottle。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "yellow":
        lower = np.array([15, 100, 100], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "blue":
        if pattern == "target":
            lower = np.array([100, 80, 80])
            upper = np.array([140, 255, 255])
        elif pattern == "line":
            lower = np.array([105, 80, 80])
            upper = np.array([135, 255, 255])
        else: # bottle or 未指定
            lower = np.array([90, 60, 40])
            upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "red":
        if pattern == "line":
            lower1 = np.array([0, 120, 70], dtype=np.uint8)
            upper1 = np.array([10, 255, 255], dtype=np.uint8)
            lower2 = np.array([170, 120, 70], dtype=np.uint8)
            upper2 = np.array([180, 255, 255], dtype=np.uint8)
        elif pattern == "target":
            lower1 = np.array([0, 90, 60], dtype=np.uint8)
            upper1 = np.array([15, 255, 210], dtype=np.uint8)
            lower2 = np.array([175, 90, 60], dtype=np.uint8)
            upper2 = np.array([180, 255, 210], dtype=np.uint8)
        else: # bottle or 未指定
            lower1 = np.array([0, 100, 100], dtype=np.uint8)
            upper1 = np.array([10, 255, 255], dtype=np.uint8)
            lower2 = np.array([170, 100, 100], dtype=np.uint8)
            upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return mask
