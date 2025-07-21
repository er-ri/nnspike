


import math

import cv2
import numpy as np

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


def get_line_edges_at_y(image, roi, target_y, threshold_value=50):
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


def find_bottle_center_with_yellow_count(image):
    # backupの内容と同じなので変更不要
    """
    Find the center coordinates and yellow pixel count of a bottle in an image using OpenCV.

    This function is optimized for real-time applications with the following improvements:
    - Accepts numpy array input instead of file paths for real-time processing
    - Uses adaptive thresholding for better edge detection under various lighting conditions
    - Applies contour area filtering to reduce noise and false detections
    - Includes aspect ratio validation to ensure bottle-like shapes
    - Uses smaller morphological kernels for better performance
    - Removes debug print statements for cleaner real-time operation

    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format)

    Returns:
        tuple: ((x, y), size, yellow_pixel_count) where (x, y) is the center coordinates,
               size is the area of the largest contour, and yellow_pixel_count is the
               number of detected yellow pixels. Returns (None, None, 0) if not found.
    """
    # Check if image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return None, None, 0

    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Color-based detection (for bottles with distinctive colors)
    # Define bottle color range (adjust based on bottle color)
    # For yellow liquid inside bottle (same thresholds as detect_color_bottle in camera.py)
    lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    # Create mask for yellow
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # type: ignore[arg-type]

    # Calculate yellow pixel count
    yellow_pixel_count = cv2.countNonZero(yellow_mask)

    # Method 2: Edge detection for bottle contours
    # Use adaptive thresholding for better edge detection under various lighting
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)  # Invert to make edges white

    # Combine color and edge information
    combined_mask = cv2.bitwise_or(yellow_mask, edges)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for real-time performance
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, yellow_pixel_count

    # Filter contours by area to remove noise (adjust minimum area as needed)
    min_area = 500  # Minimum area threshold for real-time filtering
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None, yellow_pixel_count

    # Find the largest contour (assume it's the bottle)
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # Calculate the size (area) of the largest contour
    contour_size = cv2.contourArea(largest_contour)

    # Additional validation: Check aspect ratio of contour to ensure bottle-like shape
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0

    # Bottles are typically taller than they are wide (aspect ratio > 1)
    if aspect_ratio < 0.8:  # Adjust threshold as needed
        return None, None, yellow_pixel_count

    # Calculate center using moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size, yellow_pixel_count

    return None, None, yellow_pixel_count


def find_bottle_center_with_red_count(image):
    """
    Find the center coordinates and red pixel count of a bottle in an image using OpenCV.

    この関数はfind_bottle_center_with_yellow_countの赤色版です。
    赤色領域の検出・面積・中心座標・赤ピクセル数を返します。

    Args:
        image (numpy.ndarray): 入力画像 (BGR形式)

    Returns:
        tuple: ((x, y), size, red_pixel_count)  (中心座標, 輪郭面積, 赤ピクセル数)。見つからなければ (None, None, 0)
    """
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return None, None, 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ノートブックでの解析結果を反映した赤色範囲（例: H:0-12, 170-180, S:90-, V:60-）
    lower_red1 = np.array([0, 90, 60], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 90, 60], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixel_count = cv2.countNonZero(red_mask)

    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)
    combined_mask = cv2.bitwise_or(red_mask, edges)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, red_pixel_count

    min_area = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None, None, red_pixel_count

    largest_contour = max(valid_contours, key=cv2.contourArea)
    contour_size = cv2.contourArea(largest_contour)

    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0
    if aspect_ratio < 0.8:
        return None, None, red_pixel_count

    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size, red_pixel_count

    return None, None, red_pixel_count


def find_bottle_center_with_blue_count(image):
    # backupの内容で完全修復
    """
    Find the center coordinates and blue pixel count of a bottle in an image using OpenCV.

    This function is optimized for real-time applications with the following improvements:
    - Accepts numpy array input instead of file paths for real-time processing
    - Uses adaptive thresholding for better edge detection under various lighting conditions
    - Applies contour area filtering to reduce noise and false detections
    - Includes aspect ratio validation to ensure bottle-like shapes
    - Uses smaller morphological kernels for better performance
    - Removes debug print statements for cleaner real-time operation

    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format)

    Returns:
        tuple: ((x, y), size, blue_pixel_count) where (x, y) is the center coordinates,
               size is the area of the largest contour, and blue_pixel_count is the
               number of detected blue pixels. Returns (None, None, 0) if not found.
    """
    # Check if image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return None, None, 0

    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Color-based detection (for bottles with distinctive colors)
    # Define bottle color range (adjust based on bottle color)
    # For blue liquid inside bottle (same thresholds as detect_color_bottle in camera.py)
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([130, 255, 255])

    # Create mask for blue
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)  # type: ignore[arg-type]

    # Calculate blue pixel count
    blue_pixel_count = cv2.countNonZero(blue_mask)

    # Method 2: Edge detection for bottle contours
    # Use adaptive thresholding for better edge detection under various lighting
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)  # Invert to make edges white

    # Combine color and edge information
    combined_mask = cv2.bitwise_or(blue_mask, edges)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for real-time performance
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, blue_pixel_count

    # Filter contours by area to remove noise (adjust minimum area as needed)
    min_area = 500  # Minimum area threshold for real-time filtering
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None, blue_pixel_count

    # Find the largest contour (assume it's the bottle)
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # Calculate the size (area) of the largest contour
    contour_size = cv2.contourArea(largest_contour)

    # Additional validation: Check aspect ratio of contour to ensure bottle-like shape
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0

    # Bottles are typically taller than they are wide (aspect ratio > 1)
    if aspect_ratio < 0.8:  # Adjust threshold as needed
        return None, None, blue_pixel_count

    # Calculate center using moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size, blue_pixel_count

    return None, None, blue_pixel_count


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


def get_virtual_line_edges_at_y(img, target_y, line_width=10, image_width=640, fallback_center_x=None, previous_center_x=None):
    """
    輪郭位置情報基準進路決定システム
    
    輪郭検出と軌道安定化に基づく仮想ライン中心座標取得関数。
    画像内の輪郭を解析し、検出された障害物を通過する最適経路を計算し、
    軌道の安定性を維持します。
    
    【実装要件】
    1. 輪郭検出：適応的二値化による高精度輪郭抽出
    2. 領域フィルタリング：面積・サイズ・位置による有効領域選別
    3. 除外範囲：Y≥350 AND 120≤X≤520 エリアの輪郭を除外
    4. 近接統合：距離50px以内の小輪郭を統合処理
    5. 軌道安定化：前回中心座標から±50px範囲内での軌道制限
    6. 経路選択：奥輪郭間隙→全輪郭最適間隙→単一輪郭回避の優先順位
    7. フォールバック：軌道計算失敗時の前回軌道維持または中央復帰
    
    パラメータ:
    - img: 入力画像 (BGR形式)
    - target_y: ライン中心を検出するY座標
    - line_width: 境界チェック用の幅 (デフォルト: 10)
               ※画像端からline_width//2以上離れた位置に軌道を制限
    - image_width: 画像の幅 (デフォルト: 640)
    - fallback_center_x: フォールバック中心X位置 (デフォルト: image_width // 2)
    - previous_center_x: 軌道安定化用の前回中心X座標 (デフォルト: None)
                        ※軌道安定化を有効にする場合は前回の結果を設定
    
    戻り値:
    - target_x: 最適軌道の中心X座標
    """
    
    if fallback_center_x is None:
        fallback_center_x = image_width // 2
    
    # 軌道安定化: previous_center_xが有効な場合は優先維持
    if previous_center_x is not None:
        # 前回軌道から大きく変化しない範囲で制限
        valid_center_min = max(50, previous_center_x - 50)
        valid_center_max = min(image_width - 50, previous_center_x + 50)
    else:
        valid_center_min = 50
        valid_center_max = image_width - 50
    
    # 要件: 輪郭の位置情報を取得
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bin_cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_noise)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    bin_dilated = cv2.dilate(bin_cleaned, kernel_dilate, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    bin_final = cv2.morphologyEx(bin_dilated, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(bin_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 要件: 輪郭位置情報フィルタリング（有効領域抽出）
    detected_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (50 < area < 20000 and w >= 10 and h >= 10 and y <= 450 and not (y >= 350 and 120 <= x <= 520)):
            detected_regions.append({
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
                'center': (x + w//2, y + h//2)
            })
    
    # 要件: 近接輪郭統合処理
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
            if distance < 50 and (region['area'] < 500 or other['area'] < 500):
                current_group.append(other)
                processed.add(j)
        
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            min_x = min(r['x'] for r in current_group)
            min_y = min(r['y'] for r in current_group)
            max_x = max(r['x'] + r['w'] for r in current_group)
            max_y = max(r['y'] + r['h'] for r in current_group)
            merged.append({
                'x': min_x, 'y': min_y, 'w': max_x - min_x, 'h': max_y - min_y,
                'area': sum(r['area'] for r in current_group),
                'center': ((min_x + max_x) // 2, (min_y + max_y) // 2)
            })
    detected_regions = merged
    
    # 軌道計算: 安定化優先版
    trajectory_center_x = None
    min_safe_gap = 40
    
    # 輪郭無し: 前回軌道維持または中央
    if not detected_regions:
        trajectory_center_x = previous_center_x if previous_center_x is not None else fallback_center_x
    else:
        regions_sorted = sorted(detected_regions, key=lambda x: x['x'])
        back_regions = [r for r in detected_regions if r['y'] <= target_y - 50]
        
        # 奥輪郭2個以上: 間隙通過
        if len(back_regions) >= 2:
            back_sorted = sorted(back_regions, key=lambda x: x['x'])
            left_edge = back_sorted[0]['x'] + back_sorted[0]['w']
            right_edge = back_sorted[1]['x']
            gap_width = right_edge - left_edge
            if gap_width >= min_safe_gap:
                candidate_x = (left_edge + right_edge) // 2
                # 軌道安定化: 有効範囲内かチェック
                if valid_center_min <= candidate_x <= valid_center_max:
                    trajectory_center_x = candidate_x
        
        # 全輪郭最適経路選択
        if trajectory_center_x is None and len(regions_sorted) >= 2:
            best_gap = None
            for i in range(len(regions_sorted) - 1):
                left_edge = regions_sorted[i]['x'] + regions_sorted[i]['w']
                right_edge = regions_sorted[i + 1]['x']
                gap_width = right_edge - left_edge
                if gap_width >= min_safe_gap:
                    candidate_x = (left_edge + right_edge) // 2
                    # 軌道安定化: 有効範囲内かチェック
                    if valid_center_min <= candidate_x <= valid_center_max:
                        if best_gap is None or gap_width > best_gap['width']:
                            best_gap = {'width': gap_width, 'center': candidate_x}
            
            if best_gap:
                trajectory_center_x = best_gap['center']
        
        # 単一輪郭回避
        if trajectory_center_x is None and len(detected_regions) == 1:
            region = detected_regions[0]
            contour_center = region['center'][0]
            image_center = image_width // 2
            safety_distance = 20
            
            # 反対方向回避
            if contour_center < image_center:
                candidate_x = contour_center + region['w']//2 + safety_distance
            else:
                candidate_x = contour_center - region['w']//2 - safety_distance
            
            # 軌道安定化: 有効範囲内かチェック
            if valid_center_min <= candidate_x <= valid_center_max:
                trajectory_center_x = candidate_x
        
        # フォールバック: 前回軌道維持または中央
        if trajectory_center_x is None:
            trajectory_center_x = previous_center_x if previous_center_x is not None else fallback_center_x
    
    # 最終調整：画像境界チェック
    # line_width//2 以上かつ image_width - line_width//2 - 1 以下に制限
    # これにより軌道が画像端から一定距離を保ち、ロボットの安全な制御範囲内に収める
    trajectory_center_x = max(line_width//2, min(image_width - line_width//2 - 1, trajectory_center_x))
    
    return trajectory_center_x
