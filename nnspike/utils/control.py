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
                    if 0.2 < ratio < 5.0 and major > 5 and minor > 3:
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

def get_virtual_line_edges_at_y(img, target_y, line_width=10, image_width=640, fallback_center_x=None, previous_center_x=None, preference='right'):
    """
    輪郭位置情報基準進路決定システム
    backup/20250724/control.pyの内容をそのまま追加
    """
    # --- 進路決定パラメータの初期化 ---
    if preference is None:
        preference = 'right'  # デフォルトは右優先
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
        if (50 < area < 20000 and w >= 15 and h >= 15 and y <= 450 and not (y >= 350 and 120 <= x <= 520)):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [cnt], 255)
            region_pixels = gray[mask == 255]
            avg_darkness = 255 - np.mean(region_pixels) if len(region_pixels) > 0 else 0
            # --- 極端に横長かつ暗い領域を除外 ---
            if w / h > 10 and avg_darkness > 100:
                continue
            detected_regions.append({
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
                'center': (x + w//2, y + h//2),
                'darkness': avg_darkness,
                'priority': avg_darkness * (area / 1000)
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

    # --- 進路中心計算ロジック ---
    trajectory_center_x = None
    min_safe_gap = 40  # 最小安全幅
    if not detected_regions:
        # 領域がなければ前回値または中央
        trajectory_center_x = previous_center_x if previous_center_x is not None else fallback_center_x
    else:
        regions_sorted = sorted(detected_regions, key=lambda x: x['x'])  # x座標順
        regions_by_priority = sorted(detected_regions, key=lambda x: x['priority'], reverse=True)  # 優先度順
        back_regions = [r for r in detected_regions if r['y'] <= target_y - 50]  # 奥側領域
        high_priority_threshold = 150
        high_priority_regions = [r for r in detected_regions if r['darkness'] > high_priority_threshold]
        if len(back_regions) >= 2:
            # 奥側2領域から中心に近いペアを選び、preferenceで左右決定
            img_center_x = image_width // 2
            regions_by_center = sorted(back_regions, key=lambda r: abs(r['center'][0] - img_center_x))
            pair = regions_by_center[:2]
            if preference == 'right':
                if pair[0]['center'][0] > pair[1]['center'][0]:
                    right = pair[0]
                    left = pair[1]
                else:
                    right = pair[1]
                    left = pair[0]
            elif preference == 'left':
                if pair[0]['center'][0] < pair[1]['center'][0]:
                    left = pair[0]
                    right = pair[1]
                else:
                    left = pair[1]
                    right = pair[0]
            left_edge = left['x'] + left['w']
            right_edge = right['x']
            safety_margin = 0
            if left['darkness'] > high_priority_threshold:
                safety_margin += 15
            if right['darkness'] > high_priority_threshold:
                safety_margin += 15
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
                safety_margin = 0
                if left_region['darkness'] > high_priority_threshold:
                    safety_margin += 15
                if right_region['darkness'] > high_priority_threshold:
                    safety_margin += 15
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
        # 単一領域時の中心決定
        if trajectory_center_x is None and len(detected_regions) == 1:
            region = detected_regions[0]
            contour_center = region['center'][0]
            image_center = image_width // 2
            base_safety_distance = 20
            darkness_bonus = min(region['darkness'] / 10, 30)
            safety_distance = base_safety_distance + darkness_bonus
            if contour_center < image_center:
                candidate_x = contour_center + region['w']//2 + safety_distance
            else:
                candidate_x = contour_center - region['w']//2 - safety_distance
            if valid_center_min <= candidate_x <= valid_center_max:
                trajectory_center_x = candidate_x
        # どれにも該当しない場合は前回値または中央
        if trajectory_center_x is None:
            trajectory_center_x = previous_center_x if previous_center_x is not None else fallback_center_x

    # --- 最終的な中心値を画像範囲内にクリップ ---
    trajectory_center_x = max(line_width//2, min(image_width - line_width//2 - 1, trajectory_center_x))
    return trajectory_center_x

# ヒットしたy座標（下端）とその左右端x座標を返す関数

def get_line_trace_edges_at_x320(img):
    """
    x=320の縦線上で下から上に黒色または青色ラインを探索し、
    一番下でヒットしたy座標のみを返す。
    黒: RGBすべてblack_thresh未満, 青: HSVでinRange（lower_blue=[100,80,80], upper_blue=[140,255,255]でノートブックと完全一致）
    Returns: target_y or None
    """
    import cv2
    import numpy as np
    center_x = 320
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    # 黒色マスク（固定値50）
    black_mask = np.all(img_rgb < 50, axis=2).astype(np.uint8) * 255
    # 青色マスク（引数でなく固定値）
    blue_hsv_lower = (100, 80, 80)
    blue_hsv_upper = (140, 255, 255)
    blue_mask = cv2.inRange(img_hsv, np.array(blue_hsv_lower), np.array(blue_hsv_upper))
    # 論理和
    target_mask = cv2.bitwise_or(black_mask, blue_mask)
    # x=320の縦ライン上でヒットしたy座標（下から上）
    col_target = target_mask[:, center_x]
    hit_ys = np.where(col_target == 255)[0]
    if len(hit_ys) == 0:
        return None
    target_y = int(hit_ys[-1])  # 一番下のヒット点
    return target_y

