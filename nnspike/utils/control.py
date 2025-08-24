import math
from typing import Optional, Tuple
import cv2
import numpy as np

def get_line_edges_at_y(image, roi, target_y, threshold=80) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    指定Y座標で黒ラインの左右端点（X座標）と幅を検出する。
    
    パラメータ:
        image (np.ndarray): 入力画像（BGRまたはグレースケール）
        roi (tuple): ROI（x1, y1, x2, y2）画像全体基準
        target_y (int): 検出するY座標（画像全体基準）
        threshold (int): 二値化閾値（デフォルト: 80）
    前処理:
        グレースケール化→ガウシアンブラー→二値化（binary_inv）→ノイズ除去→ROI抽出
    戻り値:
        left_x (float or None): 左端X座標
        right_x (float or None): 右端X座標
        line_width (float or None): ライン幅
    """

    # 画像がNoneまたは空の場合はNone返却
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return None, None, None
    x1, y1, x2, y2 = roi  # ROI座標
    # 前処理（詳細は引数で指定）
    if target_y < y1 or target_y >= y2:
        return None, None, None  # ROI外の場合はNone

    mask_full = control_preprocess_image(
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
    binary = mask_full[y1 : y2, x1 : x2]  # ROI抽出
    roi_row = target_y - y1  # ROI内のY座標
    if roi_row >= 0 and roi_row < (y2 - y1):
        row_data = binary[roi_row, :]
        white_pixels = np.where(row_data == 255)[0]  # 白画素の抽出
        if len(white_pixels) > 0:
            left_x_roi = white_pixels[0]
            right_x_roi = white_pixels[-1]
            left_x = x1 + left_x_roi
            right_x = x1 + right_x_roi
            line_width = right_x - left_x + 1
            return left_x, right_x, line_width
    return None, None, None  # ラインが検出できない場合

def find_bottle_center(img, color, min_area: int = 500) -> Tuple[Optional[Tuple[float, float]], Optional[float], int]:
    """
    指定色（yellow, blue, red）の物体中心座標・面積・色ピクセル数を返す。
    
    パラメータ:
        img (np.ndarray): 入力画像（BGR）
        color (str): 検出色（'yellow', 'blue', 'red'）
        min_area (int): 輪郭面積の最小値（デフォルト500）
    前処理:
        HSVマスク→メディアンブラー→ノイズ除去
    戻り値:
        center (tuple or None): 物体中心座標 (x, y)
        area (float or None): 面積
        color_pixel_count (int): 色ピクセル数
    例外:
        ValueError: colorが未対応の場合
    """

    if color not in ["yellow", "blue", "red"]:
        return None, None, 0
    if img is None or img.size == 0:
        return None, None, 0
    color_mask = get_color_mask(img, color, pattern="bottle")
    color_pixel_count = cv2.countNonZero(color_mask)
    bottle_mask = control_preprocess_image(
        color_mask,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=7,
        binarize_mode=None,
        noise_removal=["close7x7"]
    )
    contours, _ = cv2.findContours(bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, color_pixel_count
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        M = cv2.moments(contour)
        m00 = M["m00"]
        if area < min_area:
            continue
        if aspect_ratio < 0.8:
            continue
        if m00 == 0:
            continue
        if area > max_area:
            max_area = area
            best_contour = contour
    if best_contour is None:
        return None, None, color_pixel_count
    M = cv2.moments(best_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), max_area, color_pixel_count

def calculate_attitude_angle(
    offset_pixels: float,
    roi_bottom_y: int,
    camera_height: float = 0.20,
    focal_length_pixels: float = 640,
) -> float:
    """
    ピクセルオフセットからカメラ幾何で姿勢角（theta）を算出。
    画像中心からの横方向オフセットを実世界の角度に変換。
    パラメータ:
        offset_pixels (float): 画像中心からの横方向オフセット（ピクセル）
        roi_bottom_y (int): ROI下端y座標
        camera_height (float, optional): カメラ高さ[m]（デフォルト0.20）
        focal_length_pixels (float, optional): 焦点距離[px]（デフォルト640）
    戻り値:
        float: 姿勢角（theta, ラジアン）。右が正、左が負。
    備考:
        カメラパラメータはロボットごとに要調整。
    """
    image_height = 480  # 標準カメラ解像度
    ground_distance = camera_height * focal_length_pixels / (image_height - roi_bottom_y)
    lateral_offset_meters = offset_pixels * ground_distance / focal_length_pixels
    theta = math.atan2(lateral_offset_meters, ground_distance)
    return theta

# --- backup/20250724/control.pyより ---
def find_blue_target_center(img) -> Tuple[Optional[Tuple[int, int]], Optional[float], int]:
    """
    青い的（楕円）の中心座標・面積・青ピクセル数を返す。
    パラメータ:
        img (np.ndarray): BGR画像
    戻り値:
        center (tuple or None): (x, y)またはNone
        area (float or None): 面積
        blue_pixel_count (int): 青ピクセル数
    """
    # 画像がNoneまたは空の場合はNone返却
    if img is None or img.size == 0:
        return None, None, 0
    mask_blue = get_color_mask(img, "blue", pattern="target")  # 青色抽出
    mask_blue = control_preprocess_image(
        mask_blue,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=7,
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
                    if 0.2 < ratio < 2.0 and major > 5 and minor > 3:
                        if area > max_blue_area:
                            best_blue_ellipse = ellipse
                            max_blue_area = area
                            best_center = (int(cx), int(cy))
                except:
                    continue
    blue_pixel_count = cv2.countNonZero(mask_blue)
    if best_blue_ellipse is not None:
        return best_center, max_blue_area, blue_pixel_count  # 中心座標・面積・青ピクセル数
    return None, None, 0

def get_is_blue_line_at_y(img, target_y=470, min_run=30) -> bool:
    """
    指定したy座標（target_y）で、HSV条件に合致する青ピクセルがmin_run個以上連続していればTrue、そうでなければFalseを返す。
    画像全体のx方向を横断して判定する。ノイズ除去や細いラインの検出に有効。

    パラメータ:
        img (np.ndarray): BGR画像
        target_y (int): 判定するy座標（画像全体基準）
        min_run (int, optional): 青ピクセルの最小連続数（デフォルト30）。

    戻り値:
        bool: min_run個以上連続した青ピクセルがあればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or img.size == 0:
        return False
    if not (0 <= target_y < img.shape[0]):
        return False
    # 青色ラインマスク生成（HSV抽出）
    mask = get_color_mask(img, "blue", pattern="line")
    line_mask = mask[target_y, :]
    # 連続する青画素数がmin_run以上か判定
    max_run = 0
    current_run = 0
    for v in line_mask:
        if v:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run >= min_run

def get_blue_line_pixel(img) -> int:
    """
    青物体検知・面積判定・ROIクロップ・最大面積物体のみ返す。
    
    パラメータ:
        img (np.ndarray): 入力画像（BGR）
    前処理:
        青色マスク→メディアンブラー→ノイズ除去→ROI抽出
    戻り値:
        max_area (float): 最大面積物体の面積（なければ0）
    """
    # 画像がNoneまたは空の場合は0返却
    if img is None or img.size == 0:
        return 0
    _roi = (100, 200, 540, 480)  # ROI座標
    x1, y1, x2, y2 = _roi
    # 青色ラインマスク生成（HSV抽出）
    mask_full = get_color_mask(img, "blue", pattern="line")
    # メディアンブラー＋ノイズ除去
    mask_full = control_preprocess_image(
        mask_full,
        use_hsv=False,
        grayscale=False,
        clahe=False,
        blur_type="median",
        blur_ksize=7,
        binarize_mode=None,
        noise_removal=["dilate", "close7x7"]
    )
    # ROI適用
    mask_full = mask_full[y1:y2, x1:x2]
    # 輪郭検出
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        # 面積300以上かつ最大面積のみ返却
        if area > 300 and area > max_area:
            max_area = area
    return int(max_area)

# x=320の中心ラインが青的（青い楕円）にヒットしたらTrueを返す関数
def is_x320_on_blue_target(img, x_tolerance=60) -> bool:
    """
    画像内の青的（楕円）の中心がx=320±x_toleranceの範囲にあればTrueを返す。
    青的が見つからなければFalse。
    パラメータ:
        img (np.ndarray): BGR画像
        x_tolerance (int): 許容するx方向の誤差幅（ピクセル）
    戻り値:
        bool: x=320付近に青的があればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or img.size == 0:
        return False
    # 色抽出はget_color_maskで統一
    mask_blue = get_color_mask(img, "blue", pattern="target")
    # クロージング（5x5楕円カーネル）＋メディアンブラー
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
def is_x320_on_red_target(img, x_tolerance=60) -> bool:
    """
    画像内の赤的（楕円）の中心がx=320±x_toleranceの範囲にあればTrueを返す。
    赤的が見つからなければFalse。
    パラメータ:
        img (np.ndarray): BGR画像
        x_tolerance (int): 許容するx方向の誤差幅（ピクセル）
    戻り値:
        bool: x=320付近に赤的があればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or img.size == 0:
        return False
    # 色抽出はget_color_maskで統一
    mask_red = get_color_mask(img, "red", pattern="target")
    # クロージング（5x5楕円カーネル）＋メディアンブラー
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

def get_red_target_center_x(img) -> Optional[int]:
    """
    画像内の赤的（楕円）の中心x座標を返す。
    赤的が見つからなければNoneを返す。
    パラメータ:
        img (np.ndarray): BGR画像
    戻り値:
        int or None: 赤的の中心x座標、見つからなければNone
    """
    # 画像がNoneまたは空の場合はNone返却
    if img is None or img.size == 0:
        return None
    # 色抽出はget_color_maskで統一
    mask_red = get_color_mask(img, "red", pattern="target")
    # クロージング（5x5楕円カーネル）＋メディアンブラー
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
def is_left_black_line_detected(img, course) -> bool:
    """
    左側黒ラインの長さ・位置で検出する。
    パラメータ:
        img (np.ndarray): 入力画像（BGR）
        course (str): 'left'の場合は左右反転
    前処理:
        グレースケール化→CLAHE→メディアンブラー→二値化（binary_inv）→ノイズ除去→ROI抽出
    戻り値:
        bool: 条件を満たす黒ラインが検出されればTrue、なければFalse
    例外:
        FileNotFoundError: 画像がNoneの場合
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or (hasattr(img, 'size') and img.size == 0):
        return False
    _min_width = 60
    _min_height = 150
    _min_aspect = 2
    _min_area = 8000
    _roi = (0, 80, 140, 420)
    x1, y1, x2, y2 = _roi
    # leftコース時は左右反転
    if course == 'left':
        img = cv2.flip(img, 1)
    # 緑領域を白で塗りつぶし
    img = fill_green_with_white(img)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
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
    mask_roi = np.zeros_like(mask_full)
    mask_roi[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = h / (w + 1e-5)
        # ROI内で幅・高さ・アスペクト比・面積で判定
        if w >= _min_width and h >= _min_height and aspect >= _min_aspect and area >= _min_area:
            return True
    return False

def is_general_horizontal_line_detected(img) -> bool:
    """
    x=320と交差する一般的な水平黒ラインが検出されたらTrueを返す関数
    ROI: y0から540まで全体、x=320との交差必須、90度に近い角度を重視

    パラメータ:
        img (np.ndarray): BGR画像

    戻り値:
        bool: x=320と交差し90度に近い水平黒ラインが検出されればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or (hasattr(img, 'size') and img.size == 0):
        return False
    
    # 判定パラメータ
    _min_width = 150      # 幅条件
    _min_height = 10      # 高さ条件
    _max_aspect = 0.2     # アスペクト比（高さ/幅）
    _min_area = 3000      # 面積条件
    _angle_binarize_value = 10  # 角度許容範囲（0度±10または90度±10）
    _center_x = 320       # 画像中心x座標
    _roi = (200, 50, 440, 540)  # ROI（x1, y1, x2, y2）
    x1, y1, x2, y2 = _roi

    # 緑領域を白で塗りつぶし
    img = fill_green_with_white(img)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
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
    mask_roi = np.zeros_like(mask_full)
    mask_roi[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = h / w if w > 0 else float('inf')
        # 角度計算
        angle = None
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            angle_raw = rect[2]
            # minAreaRectの仕様: -90〜0度
            if angle_raw < -45:
                angle_norm = 90 + angle_raw  # 水平0付近、垂直90付近
            else:
                angle_norm = angle_raw  # 水平0付近
            angle_from_0 = abs(angle_norm)
            angle_from_90 = abs(abs(angle_norm) - 90)
        else:
            angle_from_0 = 0
            angle_from_90 = 90
        # x=320との交差判定
        crosses_center = (x <= _center_x <= x + w)
        # 0度±10または90度±10を許容
        angle_ok = (angle_from_0 <= _angle_binarize_value) or (angle_from_90 <= _angle_binarize_value)
        if (
            w >= _min_width and 
            h >= _min_height and 
            aspect_ratio <= _max_aspect and 
            area >= _min_area and 
            crosses_center and
            angle_ok
        ):
            return True
    return False

def is_horizontal_black_line_detected(img, intersection_y=450, roi=(100, 300, 540, 540)) -> bool:
    """
    x=320を通り、指定されたy座標と交差する水平黒ラインが検出されたらTrueを返す関数
    frame_1909を未検出、frame_1910を検出するようにバランス調整された実装

    パラメータ:
        img (np.ndarray): BGR画像
        intersection_y (int): 交差判定するy座標（デフォルト450）
        roi (tuple): ROI（x1, y1, x2, y2）デフォルト(100, 300, 540, 540)

    戻り値:
        bool: x=320を通り、指定されたy座標と交差する水平黒ラインが検出されればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or (hasattr(img, 'size') and img.size == 0):
        return False
    
    # 判定パラメータ
    _min_width = 400      # 幅条件
    _min_height = 50     # 高さ条件
    _max_aspect = 0.4    # アスペクト比
    _min_area = 23000    # 面積条件
    _center_x = 320      # 画像中心x座標
    x1, y1, x2, y2 = roi

    # 緑領域を白で塗りつぶし
    #img = fill_green_with_white(img)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
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
    mask_roi = np.zeros_like(mask_full)
    mask_roi[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = h / w if w > 0 else float('inf')
        crosses_center = (x <= _center_x <= x + w)
        crosses_intersection_y = (y <= intersection_y <= y + h)

        # y座標交差なら無条件で検出
        if crosses_intersection_y:
            return True
        else:
            # 通常条件
            if (w >= _min_width and h >= _min_height and aspect_ratio <= _max_aspect and area >= _min_area and crosses_center):
                return True
    return False

def is_vertical_black_line_detected(img, roi=(200, 200, 440, 540), center_tolerance=80) -> bool:
    """
    x=320±60px付近を通る縦長（垂直）黒ラインが検出されたらTrueを返す関数
    ROI: roi引数で指定した範囲で判定（デフォルト: (200, 200, 440, 540)）
    幅・高さ・アスペクト比（縦長）・面積・中心付近（x=320±60px）を重視

    パラメータ:
        img (np.ndarray): BGR画像
        roi (tuple): ROI（x1, y1, x2, y2）

    戻り値:
        bool: x=320±60px付近を通る縦長黒ラインが検出されればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if img is None or (hasattr(img, 'size') and img.size == 0):
        return False
    
    # 判定パラメータ
    _min_width = 50      # 幅条件
    _min_height = 200   # 高さ条件
    _min_aspect = 1.8   # アスペクト比
    _min_area = 11000   # 面積条件
    _center_x = 320     # 画像中心x座標
    x1, y1, x2, y2 = roi

    # 緑領域を白で塗りつぶし
    img = fill_green_with_white(img)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
        img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=3.0,
        blur_type="median",
        blur_ksize=7,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=["close7x7"]
    )
    # ROI適用
    mask_roi = np.zeros_like(mask_full)
    mask_roi[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = h / w if w > 0 else float('inf')
        # x=320±center_toleranceを通るか
        line_center_x = x + w // 2
        crosses_center = abs(line_center_x - _center_x) <= center_tolerance
        if (
            w >= _min_width and 
            h >= _min_height and 
            aspect_ratio >= _min_aspect and 
            area >= _min_area and 
            crosses_center
        ):
            return True
    return False

def get_virtual_line_target_x(img, previous_center_x=None) -> int:
    # 画像がNoneまたは空の場合は320返却
    if img is None or (hasattr(img, 'size') and img.size == 0):
        return 320
    # 仮想ライン検出用ROI座標
    _roi = (100, 100, 540, 330)
    x1, y1, x2, y2 = _roi
    mask_full = control_preprocess_image(
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
    mask_full = mask_full[y1:y2, x1:x2]
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    max_aspect = 5.0
    filtered_contours = []
    rect_centers_x = []
    rect_centers_y = []
    rects = []
    short_axes = []
    # 面積・アスペクト比でフィルタ
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h if h > 0 else 0
        if area < min_area:
            continue
        if aspect > max_aspect:
            continue
        if area >= 12000:
            continue
        cx = x + w // 2
        cy = y + h // 2
        filtered_contours.append(cnt)
        rect_centers_x.append(cx)
        rect_centers_y.append(cy)
        rects.append((x, y, w, h))
        short_axes.append(min(w, h))

    candidates = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x1 + x + w // 2
        center_y = y1 + y + h // 2
        left_edge_x = x1 + x
        right_edge_x = x1 + x + w
        area = w * h
        candidates.append({
            'center_x': center_x,
            'center_y': center_y,
            'left_edge_x': left_edge_x,
            'right_edge_x': right_edge_x,
            'area': area
        })
    candidates_sorted = sorted(candidates, key=lambda c: c['center_y'], reverse=True)
    selected = None
    for c in candidates_sorted:
        # 下から順に最初の物体を選択
        selected = c
        break
    # 物体がない場合は中央
    if selected is None:
        target_x = 320
        edge_label = "no_object"
    else:
        center_x = selected['center_x']
        left_edge_x = selected['left_edge_x']
        right_edge_x = selected['right_edge_x']

        # previous_center_xから回避方向ラベル初期化
        if previous_center_x is None or previous_center_x == 320:
            pre_edge_label = None
        elif previous_center_x < 320:
            pre_edge_label = 'right'
        else:
            pre_edge_label = 'left'

        # 320±50pxに物体があれば前回方向を優先
        any_in_center = any(270 <= c['center_x'] <= 370 for c in candidates)
        if any_in_center and pre_edge_label is not None:
            edge_label = pre_edge_label
        else:
            if center_x < 320:
                edge_label = 'right'
            else:
                edge_label = 'left'

        if edge_label == 'right':
            edge_x = right_edge_x
            target_x = edge_x + 150
        elif edge_label == 'left':
            edge_x = left_edge_x
            target_x = edge_x - 150
        else:
            # 物体中心には絶対向かわない。安全なデフォルト値。
            target_x = 320

    # previous_center_xによるジャンプ制限
    if candidates and previous_center_x is not None and 'target_x' in locals():
        max_delta = 40  # 許容する最大変化量
        if abs(target_x - previous_center_x) > max_delta:
            if target_x > previous_center_x:
                target_x = previous_center_x + max_delta
            else:
                target_x = previous_center_x - max_delta

    return target_x

def control_preprocess_image(
    img,
    use_hsv=False,         # BGR→HSV変換
    grayscale=False,       # グレースケール化
    clahe=False,           # CLAHE（コントラスト強調）
    clahe_clipLimit=3.0,   # CLAHEパラメータ
    blur_type=None,        # 平滑化フィルター
    blur_ksize=7,          # フィルターサイズ
    binarize_mode=None,    # 二値化タイプ
    binarize_value: Optional[int]=120, # 二値化閾値
    noise_removal=None,    # ノイズ除去
    ) -> np.ndarray:

    # --- 色空間変換 ---
    if use_hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- グレースケール化 ---
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- コントラスト強調（CLAHE） ---
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=(8,8))
        img = clahe_obj.apply(img)

    # --- フィルター（平滑化） ---
    if blur_type is not None:
        if blur_type == "gaussian":
            img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
        elif blur_type == "median":
            img = cv2.medianBlur(img, blur_ksize)

    # --- 二値化 ---
    if binarize_mode is not None:
        if binarize_mode == "otsu":
            # Otsu + binary_inv
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif binarize_mode == "binary":
            # binary
            if binarize_value is None:
                raise ValueError("binarize_value must be specified for binary mode")
            _, img = cv2.threshold(img, binarize_value, 255, cv2.THRESH_BINARY)
        elif binarize_mode == "binary_inv":
            # binary_inv（明示）
            if binarize_value is None:
                raise ValueError("binarize_value must be specified for binary_inv mode")
            _, img = cv2.threshold(img, binarize_value, 255, cv2.THRESH_BINARY_INV)
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
                img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
            elif nr == "dilate7x2":
                img = cv2.dilate(img, np.ones((7, 7), np.uint8), iterations=2)
            elif nr == "dilate7x3":
                img = cv2.dilate(img, np.ones((7, 7), np.uint8), iterations=3)
            elif nr == "close3x3":
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            elif nr == "close5x5_ellipse":
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            elif nr == "close7x7":
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            elif nr == "close11x11":
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            elif nr == "open3x3":
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return img

def get_color_mask(img, color, pattern=None) -> np.ndarray:
    """
    指定色のHSVマスクを返す（yellow, blue, red対応）。
    patternはbottle/line/targetのみ。未指定時はbottle。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "yellow":
        lower = np.array([15, 100, 100], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "blue":
        if pattern == "line":
            lower = np.array([95, 100, 50], dtype=np.uint8)
            upper = np.array([145, 255, 255], dtype=np.uint8)
        elif pattern == "target":
            lower = np.array([100, 80, 80], dtype=np.uint8)
            upper = np.array([140, 255, 255], dtype=np.uint8)
        else: # bottle or 未指定
            lower = np.array([90, 60, 40], dtype=np.uint8)
            upper = np.array([140, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "red":
        if pattern == "target":
            lower1 = np.array([0, 90, 60], dtype=np.uint8)
            upper1 = np.array([15, 255, 210], dtype=np.uint8)
            lower2 = np.array([175, 90, 60], dtype=np.uint8)
            upper2 = np.array([180, 255, 210], dtype=np.uint8)
        else: # bottle or 未指定
            lower1 = np.array([0, 90, 60], dtype=np.uint8)
            upper1 = np.array([12, 255, 255], dtype=np.uint8)
            lower2 = np.array([170, 90, 60], dtype=np.uint8)
            upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    return mask

# --- 緑領域を白で塗りつぶす独立メソッド ---
def fill_green_with_white(img):
    """
    画像の緑領域（HSV指定）を白で塗りつぶす。
    パラメータ:
        img (np.ndarray): BGR画像
    戻り値:
        np.ndarray: 緑領域が白で塗りつぶされた画像（BGR）
    """
    if img.ndim == 3 and img.shape[2] == 3:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 120, 60], dtype=np.uint8)
        upper_green = np.array([90, 255, 220], dtype=np.uint8)
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        img = img.copy()
        img[green_mask != 0] = [255, 255, 255]
    return img
