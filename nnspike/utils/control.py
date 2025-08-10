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
    blur_kernel=5
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
            if area > 5:
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
    for i, region in enumerate(detected_regions):
        if i in processed:
            continue
        current_group = [region]
        processed.add(i)
        for j, other in enumerate(detected_regions):
            if j in processed or j <= i:
                continue
            dx = region['center'][0] - other['center'][0]
            dy = region['center'][1] - other['center'][1]
            distance = (dx*dx + dy*dy) ** 0.5
            # 近接かつ小面積同士は1つの領域にまとめる
            if distance < 100 and (region['area'] < 1000 or other['area'] < 1000):
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
    min_safe_gap = 80  # 最小安全幅（60→80に拡大：黒色障害物対策）
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
                    safety_margin = 80  # 基本マージン（120→80に縮小）
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
                safe_distance_from_obstacle = 200  # 黒色障害物からの安全距離（200ピクセル）
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
            safety_margin = 60  # 基本安全マージン縮小（80→60：過度な回避を防ぐ）
            if left['darkness'] > high_priority_threshold:
                safety_margin += 60  # 暗い障害物：追加で60（合計120）
            if right['darkness'] > high_priority_threshold:
                safety_margin += 60  # 暗い障害物：追加で60（合計120）
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
                safety_margin = 60  # 基本安全マージン縮小（80→60：過度な回避を防ぐ）
                if left_region['darkness'] > high_priority_threshold:
                    safety_margin += 60  # 暗い障害物：追加で60（合計120）
                if right_region['darkness'] > high_priority_threshold:
                    safety_margin += 60  # 暗い障害物：追加で60（合計120）
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
            # 黒色障害物からの安全距離（普通の障害物140px、暗い障害物200px）
            safe_distance_from_obstacle = 200 if region['darkness'] > high_priority_threshold else 140
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

# ヒットしたy座標（下端）とその左右端x座標を返す関数

def get_line_trace_edges_at_x320(img):
    """
    画面を横切る超絶ロング黒ラインを検出し、
    x=320を通る一番下のライン位置を返す。
    Returns: target_y or None
    """
    import cv2
    import numpy as np
    center_x = 320
    
    # ROI固定値：画面最下部まで検出（y=500検出のため）
    roi_x_start = 100   # 左端100削る
    roi_x_end = 540     # 右端100削る（640-100=540）
    roi_y_start = 450   # y=450以下無視（より近距離検出）
    roi_y_end = 540     # 画面最下部まで（540px、十分な範囲）
    
    # ROI抽出
    roi = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # グレースケール変換 + 閾値処理（より厳格）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)  # 40→25に厳格化
    
    # 輪郭検出による超絶ロング黒ライン検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_line_y = None
    detected_lines = []
    
    for cnt in contours:
        x_roi, y_roi, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # ROI座標を元の画像座標に変換
        x = x_roi + roi_x_start
        y = y_roi + roi_y_start
        
        # ROI範囲での横ライン条件：ごく近距離の明確な黒いライン検出
        roi_width = 440  # 540-100=440 固定値
        min_width = int(roi_width * 0.5)  # ROI幅の50%以上（220px、緩い条件）
        
        # デバッグ用緩い条件で検出範囲確認：
        if (w >= min_width and 
            h >= 3 and 
            area >= 300 and 
            x <= center_x <= x + w):
            line_bottom = y + h
            detected_lines.append((w, h, area, line_bottom, x, y))  # x, y も記録
            # 最も下のライン位置を選択（bottom位置が最大のもの）
            if valid_line_y is None or line_bottom > valid_line_y:
                valid_line_y = line_bottom
    
    # デバッグ出力：ROI範囲とライン検出状況
    total_contours = len(contours)
    if total_contours > 0:
        print(f"[DEBUG] ROI範囲: x=100-540, y=450-540")
        print(f"[DEBUG] 検出された輪郭数: {total_contours}")
        
        # 検出条件の定義
        roi_width = 440  # 540-100=440
        min_width = int(roi_width * 0.5)  # 220px
        
        # 全ての輪郭の詳細チェック + 検出対象の表示
        detected_targets = []
        for i, cnt in enumerate(contours):
            x_roi, y_roi, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # ROI座標を元の画像座標に変換
            x = x_roi + roi_x_start
            y = y_roi + roi_y_start
            line_bottom = y + h
            
            # 検出対象かどうかチェック
            all_conditions_met = (w >= min_width and 
                                h >= 3 and 
                                area >= 300 and 
                                x <= center_x <= x + w)
            
            if all_conditions_met:
                detected_targets.append((line_bottom, w, h, area, x, y))
                print(f"  ✅ 検出対象 {len(detected_targets)}: bottom={line_bottom}, w={w}, h={h}, area={area}")
            else:
                print(f"  ❌ 除外: bottom={line_bottom}, w={w}, h={h}, area={area}")
        
        # 最も下の検出対象を表示
        if detected_targets:
            detected_targets.sort(key=lambda x: x[0], reverse=True)  # bottom位置で降順ソート
            best_bottom = detected_targets[0][0]
            print(f"[RESULT] 検出対象数={len(detected_targets)}, 最下位bottom={best_bottom}, selected_y={valid_line_y}")
        else:
            print(f"[RESULT] detected_long_lines=0, valid_line_y={valid_line_y}")
    
    return valid_line_y

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
                    if 0.2 < ratio < 5.0 and major > 5 and minor > 3:
                        if area > max_red_area:
                            max_red_area = area
                            best_center_x = int(cx)
                except:
                    continue
    return best_center_x

# 黒ラインの長さや位置で判定する関数（画像直接渡し、条件はプライベート変数）
def is_left_black_line_detected(img):
    _min_width = 60
    _min_height = 150
    _min_aspect = 2
    _min_area = 8000
    _roi = (0, 80, 140, 420)
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = mask.shape
    mask[:, w//2:] = 0
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=3)
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

def is_horizontal_black_line_detected(img):
    """
    画面全体を横切る水平な黒いラインが検出されたらTrueを返す関数
    
    Args:
        img: BGR画像 (numpy.ndarray)
    
    Returns:
        bool: 画面全体を横切る水平な黒いラインが検出されればTrue、なければFalse
    """
    _min_width = 400   # 画面全体を横切るための最小幅
    _min_height = 15   # 水平ラインの最小高さ
    _max_aspect = 0.3  # 水平ラインのアスペクト比上限（高さ/幅 < 0.3）
    _min_area = 5000   # 最小面積
    
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去と形状の強調
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = hh / (ww + 1e-5)  # 高さ/幅
        
        # 画面全体を横切る水平ラインの条件
        if ww >= _min_width and hh >= _min_height and aspect <= _max_aspect and area >= _min_area:
            return True
    
    return False

def is_vertical_black_line_detected(img):
    """
    画面下部まで続く垂直な黒いラインが検出されたらTrueを返す関数
    （上部は途切れていても良い、下部が画面下端まで続いていることが重要）
    
    Args:
        img: BGR画像 (numpy.ndarray)
    
    Returns:
        bool: 画面下部まで続く垂直な黒いラインが検出されればTrue、なければFalse
    """
    _min_width = 3     # 垂直ラインの最小幅（5→3に緩和）
    _min_height = 80   # 最小高さ（100→80に緩和）
    _min_aspect = 1.0  # 垂直ラインのアスペクト比下限（1.5→1.0に緩和）
    _min_area = 200    # 最小面積（500→200に緩和）
    _center_x = 320    # ロボットの中央線
    _center_tolerance = 100  # 中央からの許容範囲（±100px）
    
    if img is None:
        raise FileNotFoundError("画像がNoneです")
    
    img_height = img.shape[0]  # 画像の高さ取得
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)  # 固定閾値50
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_lines = []
    rejected_lines = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = hh / (ww + 1e-5)  # 高さ/幅
        
        # 画面下部まで続く垂直ラインの条件
        line_bottom = y + hh
        reaches_bottom = line_bottom >= img_height - 200  # 画面下端から200px以内（50→200に大幅緩和）
        
        # 各条件チェック
        width_ok = ww >= _min_width
        height_ok = hh >= _min_height  
        aspect_ok = aspect >= _min_aspect
        area_ok = area >= _min_area
        bottom_ok = reaches_bottom
        
        # 中央線との平行条件：垂直ラインの中央がx=320±100以内
        line_center_x = x + ww // 2
        parallel_ok = abs(line_center_x - _center_x) <= _center_tolerance
        
        # デバッグ：輪郭詳細を表示
        # print(f"  Contour: x={x}, center_x={line_center_x}, w={ww}, h={hh}, aspect={aspect:.2f}, area={area:.0f}, bottom={line_bottom}")
        # print(f"    Checks: width_ok={width_ok}, height_ok={height_ok}, aspect_ok={aspect_ok}, area_ok={area_ok}, bottom_ok={bottom_ok}, parallel_ok={parallel_ok}")
        
        if width_ok and height_ok and aspect_ok and area_ok and bottom_ok and parallel_ok:
            detected_lines.append((x, line_center_x, ww, hh, aspect, area, line_bottom))
        else:
            rejected_lines.append((x, line_center_x, ww, hh, aspect, area, line_bottom, width_ok, height_ok, aspect_ok, area_ok, bottom_ok, parallel_ok))
    
    # デバッグ出力
    # total_contours = len(contours)
    # detected_vertical = len(detected_lines) > 0
    # print(f"[DEBUG] is_vertical_black_line_detected: total_contours={total_contours}, detected_vertical_lines={len(detected_lines)}, img_height={img_height}")
    # print(f"  Conditions: min_width={_min_width}, min_height={_min_height}, min_aspect={_min_aspect}, min_area={_min_area}, center_tolerance=±{_center_tolerance}")
    # for i, (x, center_x, w, h, asp, area, bottom) in enumerate(detected_lines):
    #     print(f"  ✅ Parallel Line {i+1}: x={x}, center_x={center_x}, width={w}, height={h}, aspect={asp:.2f}, area={area:.0f}, bottom_y={bottom}")
    # for i, (x, center_x, w, h, asp, area, bottom, w_ok, h_ok, a_ok, ar_ok, b_ok, p_ok) in enumerate(rejected_lines[:3]):  # 最初の3つだけ
    #     print(f"  ❌ Rejected {i+1}: x={x}, center_x={center_x}, w={w}({w_ok}), h={h}({h_ok}), asp={asp:.2f}({a_ok}), area={area:.0f}({ar_ok}), bottom={bottom}({b_ok}), parallel({p_ok})")
    
    detected_vertical = len(detected_lines) > 0
    return detected_vertical
