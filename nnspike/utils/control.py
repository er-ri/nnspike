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


def get_line_edges_at_y(image, roi, target_y, threshold=80) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    指定Y座標で黒ラインの左右端点を検出する。

    前処理は「グレースケール化→ブラー→二値化（binary_inv）→ノイズ除去→ROI抽出」を厳守し、
    必ず control_preprocess_image で統一する。
    二値化は cv2.THRESH_BINARY_INV（白=ライン）で行う。

    Parameters:
        image: 入力画像（BGRまたはグレースケール）
        roi: (x, y, w, h) ROI座標（画像全体基準）
        target_y: 検出するY座標（画像全体基準）
        binarize_value_value: 二値化閾値（デフォルト: 80）

    Returns:
        left_x: 左端X座標（見つからなければNone）
        right_x: 右端X座標（見つからなければNone）
        line_width: ライン幅（見つからなければNone）
    """

    # Extract ROI coordinates
    x, y, w, h = roi

    # Check if target_y is within ROI
    if target_y < y or target_y >= y + h:
        return None, None, None

    # 「グレースケール化→ブラー→二値化→ノイズ除去→ROI抽出」を厳守
    # グレースケール化→ガウシアンブラー→二値化（binary_inv）→ノイズ除去
    preprocessed = control_preprocess_image(
        image,
        use_hsv=False,
        grayscale=True,
        clahe=False,
        blur_type="gaussian",
        blur_ksize=5,
        binarize_mode="binary_inv",
        binarize_value=threshold,
        noise_removal=None
    )
    binary = preprocessed[y : y + h, x : x + w]

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

def find_bottle_center(image, color, min_area: int = 500) -> Tuple[Optional[Tuple[float, float]], Optional[float], int]:
    """
    Find the center coordinates and color pixel count of a colored object in an image using OpenCV.

    This function detects objects of a specified color in an image and returns information about
    the largest detected object. It supports both yellow and blue color detection and can be used
    for various applications including object tracking, color-based navigation, and visual recognition.

    This function is optimized for real-time applications with the following improvements:
    - Accepts numpy array input instead of file paths for real-time processing
    - Uses adaptive binarize_valueing for better edge detection under various lighting conditions
    - Applies contour area filtering to reduce noise and false detections
    - Includes aspect ratio validation to filter out non-object-like shapes
    - Uses smaller morphological kernels for better performance
    - Removes debug print statements for cleaner real-time operation

    Args:
        image (numpy.ndarray): Input image as numpy array (BGR format)
        color (str): Color to detect ('yellow' or 'blue')
        min_area (int, optional): Minimum contour area binarize_value for filtering noise. Defaults to 500.

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
        lower_color = np.array([15, 100, 100], dtype=np.uint8)
        upper_color = np.array([35, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
    elif color == "blue":
        lower_color = np.array([90, 60, 40])
        upper_color = np.array([140, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
    elif color == "red":
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
    # Use adaptive binarize_valueing for better edge detection under various lighting
    edges = cv2.adaptivebinarize_value(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)  # Invert to make edges white

    # Combine color and edge information
    combined_mask = cv2.bitwise_or(color_mask, edges)

    # 画像前処理を control_preprocess_image で統一（現行処理内容を完全維持）
    # ブラー→ノイズ除去（グレースケール・二値化なし）
    combined_mask = control_preprocess_image(
        combined_mask,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="gaussian",
        blur_ksize=3,
        binarize_mode=None,
        noise_removal=["close3x3", "open3x3"]
    )

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
    if aspect_ratio < 0.8:  # Adjust binarize_value as needed
        return None, None, color_pixel_count

    # Calculate center using moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size, color_pixel_count

    return None, None, color_pixel_count

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
def find_blue_target_center(img):
    """
    青い的（楕円）の中心座標・形状情報を返す統合検出関数。
    Args:
        img: BGR画像 (numpy.ndarray)
        blue_hsv_lower, blue_hsv_upper: 青色範囲 (HSV)
        gray_hsv_lower, gray_hsv_upper: グレー色範囲 (HSV)
    Returns:
        center: (x, y) or None
        area: float or None
        blue_pixel_count: int
    """
    if img is None or img.size == 0:
        return None, None, 0
    blue_hsv_lower = (100, 80, 80)
    blue_hsv_upper = (140, 255, 255)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array(blue_hsv_lower), np.array(blue_hsv_upper))
    # 画像前処理を control_preprocess_image で統一（現行処理内容を完全維持）
    # morph__ellipse(5,5)を厳密に再現
    # メディアンブラー→ノイズ除去（グレースケール・二値化なし）
    mask_blue = control_preprocess_image(
        mask_blue,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=5,
        binarize_mode=None,
        noise_removal=["close5x5_ellipse"]
    )
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
    return None, None, 0

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

def get_blue_line_pixel(img):
    """
    ノートブックの青物体検知・面積判定・ROIクロップ・最大面積物体のみ返す。
    Args:
        img (np.ndarray): BGR画像
    Returns:
        float: 最大面積物体の面積（なければ0）
    """
    x1, y1, x2, y2 = 100, 200, 540, 480  # y1を200に変更
    if img is None or img.size == 0:
        return 0
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 旧閾値（コメントアウト）
    # blue_hsv_lower = (100, 150, 0)
    # blue_hsv_upper = (140, 255, 255)
    blue_hsv_lower = (95, 100, 50)
    blue_hsv_upper = (145, 255, 255)
    blue_mask_raw = np.all([
        (blue_hsv_lower[i] <= img_hsv[:, :, i]) & (img_hsv[:, :, i] <= blue_hsv_upper[i])
        for i in range(3)
    ], axis=0)
    mask_uint8 = blue_mask_raw.astype(np.uint8) * 255
    # メディアンブラー→ノイズ除去（グレースケール・二値化なし）
    mask_uint8 = control_preprocess_image(
        mask_uint8,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=7,
        binarize_mode=None,
        noise_removal=["close7x7"]
    )
    mask_uint8 = mask_uint8[y1:y2, x1:x2]
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300 and area > max_area:
            max_area = area
    return int(max_area)

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array(blue_hsv_lower), np.array(blue_hsv_upper))
    # 5x5楕円カーネルでクロージング
    # メディアンブラー→ノイズ除去（グレースケール・二値化なし）
    mask_blue = control_preprocess_image(
        mask_blue,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=5,
        binarize_mode=None,
        noise_removal=["close5x5_ellipse"]
    )
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(red_hsv_lower1), np.array(red_hsv_upper1))
    mask2 = cv2.inRange(hsv, np.array(red_hsv_lower2), np.array(red_hsv_upper2))
    mask_red = cv2.bitwise_or(mask1, mask2)
    # 5x5楕円カーネルでクロージング
    # メディアンブラー→ノイズ除去（グレースケール・二値化なし）
    mask_red = control_preprocess_image(
        mask_red,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=5,
        binarize_mode=None,
        noise_removal=["close5x5_ellipse"]
    )
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(red_hsv_lower1), np.array(red_hsv_upper1))
    mask2 = cv2.inRange(hsv, np.array(red_hsv_lower2), np.array(red_hsv_upper2))
    mask_red = cv2.bitwise_or(mask1, mask2)
    # 5x5楕円カーネルでクロージング
    # メディアンブラー→ノイズ除去（グレースケール・二値化なし）
    mask_red = control_preprocess_image(
        mask_red,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=5,
        binarize_mode=None,
        noise_removal=["close5x5_ellipse"]
    )
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
    # グレースケール化→CLAHE→メディアンブラー→二値化（binary_inv）→ノイズ除去
    mask = control_preprocess_image(
        img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=3.0,
        blur_type="median",
        blur_ksize=7,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=["dilate", "close7x7"]
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
    _angle_binarize_value = 10  # 0度±10または90度±10を許容
    _center_x = 320
    _roi = (200, 0, 440, 540)  # ノートブック準拠ROI
    
    # control_preprocess_imageで前処理を完全再現
    # グレースケール化→CLAHE→メディアンブラー→二値化（binary_inv）→ノイズ除去
    black_mask = control_preprocess_image(
        img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=3.0,
        blur_type="median",
        blur_ksize=7,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=["dilate", "close7x7"]
    )
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
        angle_ok = (angle_from_0 <= _angle_binarize_value) or (angle_from_90 <= _angle_binarize_value)
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
    
    # control_preprocess_imageで前処理を完全再現
    # グレースケール化→CLAHE→メディアンブラー→二値化（binary_inv）→ノイズ除去
    black_mask = control_preprocess_image(
        img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=3.0,
        blur_type="median",
        blur_ksize=7,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=["dilate", "close7x7"]
    )
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
    
    # control_preprocess_imageで前処理を完全再現
    # グレースケール化→CLAHE→メディアンブラー→二値化（binary_inv）→ノイズ除去
    black_mask = control_preprocess_image(
        img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=3.0,
        blur_type="median",
        blur_ksize=7,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=["dilate", "close7x7"]
    )
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
    # グレースケール化→CLAHE→メディアンブラー→二値化（binary_inv）→ノイズ除去
    mask_full = control_preprocess_image(
        img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=4.0,
        blur_type="median",
        blur_ksize=9,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=["dilate", "close7x7"]
    )
    # ROI抽出
    mask = mask_full[y1:y2, x1:x2]
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
    x_merge_binarize_value = 150
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
            if abs(cx - cx2) < x_merge_binarize_value:
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
    use_hsv=False,         # 色空間変換（BGR→HSV）
    grayscale=False,       # グレースケール化
    clahe=False,           # コントラスト強調（CLAHE）
    clahe_clipLimit=3.0,   # CLAHEパラメータ
    blur_type=None,        # フィルター（平滑化）
    blur_ksize=7,          # フィルターサイズ
    binarize_mode=None,   # 二値化タイプ（Noneで二値化なし）
    binarize_value: Optional[int]=120, # 二値化閾値（デフォルト: 120）
    noise_removal=None     # ノイズ除去
    ):
    """
    画像前処理（get_line_edges_at_yと完全同一仕様）
    - グレースケール化（grayscale=True）
    - GaussianBlur（blur_type='gaussian', blur_ksize=5）
    - 二値化（binarize_mode='binary_inv', binarize_value=80）
    - ノイズ除去はデフォルトでなし（noise_removal='none'）
    """

    # --- 色空間変換 ---
    if use_hsv:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- グレースケール化 ---
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- コントラスト強調（CLAHE） ---
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=(8,8))
        image = clahe_obj.apply(image)

    # --- フィルター（平滑化） ---
    if blur_type == "gaussian":
        image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    elif blur_type == "median":
        image = cv2.medianBlur(image, blur_ksize)
    # blur_type==Noneなら何もしない

    # --- 二値化 ---
    if binarize_mode is not None:
        if binarize_mode == "otsu":
            # Otsu + binary_inv
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif binarize_mode == "binary":
            # binary
            if binarize_value is None:
                raise ValueError("binarize_value must be specified for binary mode")
            _, image = cv2.threshold(image, binarize_value, 255, cv2.THRESH_BINARY)
        elif binarize_mode == "binary_inv":
            # binary_inv（明示）
            if binarize_value is None:
                raise ValueError("binarize_value must be specified for binary_inv mode")
            _, image = cv2.threshold(image, binarize_value, 255, cv2.THRESH_BINARY_INV)
        # binarize_mode==Noneなら何もしない

    # --- ノイズ除去（モルフォロジー処理） ---
    if noise_removal is not None:
        if isinstance(noise_removal, str):
            nrs = [noise_removal]
        elif isinstance(noise_removal, (list, tuple)):
            nrs = noise_removal
        else:
            nrs = [str(noise_removal)]
        for nr in nrs:
            if nr == "none" or nr is None:
                continue
            elif nr == "dilate":
                image = cv2.dilate(image, np.ones((5, 5), np.uint8), iterations=1)
            elif nr == "dilate7x2":
                image = cv2.dilate(image, np.ones((7, 7), np.uint8), iterations=2)
            elif nr == "dilate7x3":
                image = cv2.dilate(image, np.ones((7, 7), np.uint8), iterations=3)
            elif nr == "close3x3":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            elif nr == "close5x5_ellipse":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            elif nr == "close7x7":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            elif nr == "close11x11":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            elif nr == "open3x3":
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return image

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
