import math
from typing import Optional, Tuple
import cv2
import numpy as np
from nnspike.constants import (
    ROI_CNN,
    ROI_VIRTUAL,
    ROI_LOOP,
    ROI_LINE_LEFT,
    ROI_LINE_HORIZON1,
    ROI_LINE_HORIZON2,
    ROI_LINE_HORIZON3,
    ROI_LINE_VERTICAL1,
    ROI_LINE_CORNER,
    ROI_LINE_STRAIGHT_FAST,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)

# HSV色範囲定数（メモリ最適化：毎回のnp.array作成を回避）
_HSV_RANGES = {
    "yellow": {
        "bottle": (np.array([15, 100, 100], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
    },
    "blue": {
        "line": (np.array([95, 100, 50], dtype=np.uint8), np.array([145, 255, 255], dtype=np.uint8)),
        "target": (np.array([100, 80, 80], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8)),
        "bottle": (np.array([90, 60, 40], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
    },
    "red": {
        "target": (
            (np.array([0, 90, 60], dtype=np.uint8), np.array([15, 255, 210], dtype=np.uint8)),
            (np.array([175, 90, 60], dtype=np.uint8), np.array([180, 255, 210], dtype=np.uint8))
        ),
        "bottle": (
            (np.array([0, 90, 60], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
            (np.array([170, 90, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        )
    }
}

# 緑色範囲定数
_GREEN_RANGE = (np.array([35, 120, 60], dtype=np.uint8), np.array([90, 255, 220], dtype=np.uint8))

# ピンク色範囲定数（例：用途に応じて調整可）
_PINK_RANGE = (np.array([140, 60, 100], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8))

# モルフォロジー演算カーネル定数（事前計算でカーネル作成コストを削減）
_MORPHOLOGY_KERNELS = {
    'dilate_5x5': np.ones((5, 5), np.uint8),
    'dilate_7x7': np.ones((7, 7), np.uint8),
    'close_3x3': np.ones((3, 3), np.uint8),
    'close_7x7': np.ones((7, 7), np.uint8),
    'close_11x11': np.ones((11, 11), np.uint8),
    'ellipse_5x5': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    'open_3x3': np.ones((3, 3), np.uint8)
}

# --- offset_pixels計算用 ---
def get_offset_pixels(target_x, roi):
    """
    ROI内の中心からのX方向オフセットピクセル数を計算する
    :param target_x: 対象座標X
    :param roi: (x1, y1, x2, y2) のタプル（例: ROI_CNN）
    :return: offset_pixels (int)
    """
    x1, _, x2, _ = roi
    roi_center_x = (x2 - x1) // 2
    mx = target_x - x1
    offset_pixels = mx - roi_center_x
    return offset_pixels

def get_line_edges_at_y(image, roi, target_y, threshold_value=80) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    指定Y座標で黒ラインの左右端点（X座標）と幅を検出する。
    
    パラメータ:
    image (np.ndarray): 入力画像（BGRまたはグレースケール）
    roi (tuple): ROI（x1, y1, x2, y2）画像全体基準
    target_y (int): 検出するY座標（画像全体基準）
    threshold_value (int): 二値化閾値（デフォルト: 80）
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
        binarize_value=threshold_value,
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

def find_bottle_center(image, color, roi=ROI_CNN) -> Tuple[Optional[Tuple[int, int]], Optional[float], int]:
    """
    指定色（yellow, blue, red）の物体中心座標・面積・色ピクセル数を返す。
    roi指定時はROI内で検出し、中心座標は元画像座標で返す。
    
    パラメータ:
        image (np.ndarray): 入力画像（BGR）
        color (str): 検出色（'yellow', 'blue', 'red'）
        roi (tuple or None): ROI (x1, y1, x2, y2) 指定時はその範囲で検出
    前処理:
        HSVマスク→メディアンブラー→ノイズ除去→ROI適用
    戻り値:
        center (tuple or None): 物体中心座標 (x, y) ※元画像座標
        area (float or None): 面積
        color_pixel_count (int): 色ピクセル数
    例外:
        ValueError: colorが未対応の場合
    """

    min_area = 490  # 輪郭面積の最小値（内部定数）

    if color not in ["yellow", "blue", "red"]:
        return None, None, 0
    if image is None or image.size == 0:
        return None, None, 0
    color_mask = get_color_mask(image, color, pattern="bottle")
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
    # ROI適用（roiは必ず指定される前提）
    x1, y1, x2, y2 = roi
    mask_roi = np.zeros_like(bottle_mask)
    mask_roi[y1:y2, x1:x2] = bottle_mask[y1:y2, x1:x2]
    bottle_mask = mask_roi
    contours, _ = cv2.findContours(bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0
    max_area = 0
    best_rect = None
    color_pixel_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        # 面積・重心・色ピクセル条件
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # M = cv2.moments(contour)  # 不要：重心計算を使用していないため
        # m00 = M["m00"]            # 不要：areaと同じ値
        # if m00 == 0:              # 不要：既にarea < min_areaでフィルタ済み
        #     continue
        rect_area = w * h
        if rect_area == 0:
            continue
        ratio = area / rect_area
        # 物体面積/外接矩形面積比率が0.5以下は除外
        if ratio <= 0.5:
            continue
        # 外接矩形範囲内の色ピクセル数
        rect_mask = color_mask[y:y+h, x:x+w]
        rect_color_pixel_count = cv2.countNonZero(rect_mask)
        if area <= rect_color_pixel_count / 2:
            continue
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)
            color_pixel_count = rect_color_pixel_count
    if best_rect is None:
        # 検知対象外の場合はすべてNone/0で返す
        return None, None, 0
    # 外接矩形の中心座標
    cx = int(best_rect[0] + best_rect[2] / 2)
    cy = int(best_rect[1] + best_rect[3] / 2)
    return (cx, cy), max_area, color_pixel_count

# --- backup/20250724/control.pyより ---
def find_blue_target_center(image) -> Tuple[Optional[Tuple[int, int]], Optional[float], int]:
    """
    青い的（楕円）の中心x座標と上端y座標・面積・青ピクセル数を返す。
    パラメータ:
        image (np.ndarray): BGR画像
    戻り値:
        center (tuple or None): (x, top_y)またはNone ※yは楕円の上端座標
        area (float or None): 面積
        blue_pixel_count (int): 実際に検出された楕円の青ピクセル数（外接矩形内）
    """
    # 画像がNoneまたは空の場合はNone返却
    if image is None or image.size == 0:
        return None, None, 0
    mask_blue = get_color_mask(image, "blue", pattern="target")  # 青色抽出
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
    best_contour = None
    for cnt in contours_blue:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            ellipse = None
            try:
                ellipse = cv2.fitEllipse(cnt)
            except:
                continue
            (cx, cy), (major, minor), angle = ellipse
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            if rect_ratio < 0.4:
                continue
            if area > max_blue_area:
                best_blue_ellipse = ellipse
                max_blue_area = area
                best_contour = cnt
                # 輪郭から直接最上端Y座標を取得（最も直接的な方法）
                contour_points = cnt.reshape(-1, 2)  # 輪郭点を(N,2)形状に変換
                y_coordinates = contour_points[:, 1]  # Y座標を抽出
                min_y = np.min(y_coordinates)  # Y座標の最小値
                top_y = int(min_y.item())  # NumPy scalar を Python int に変換
                best_center = (int(cx), top_y)
    
    # 実際に検出された楕円のピクセル数を計算
    if best_blue_ellipse is not None and best_contour is not None:
        # 検出された楕円の外接矩形内の青ピクセル数を計算
        x, y, w, h = cv2.boundingRect(best_contour)
        ellipse_mask_roi = mask_blue[y:y+h, x:x+w]
        ellipse_blue_pixel_count = cv2.countNonZero(ellipse_mask_roi)
        return best_center, max_blue_area, ellipse_blue_pixel_count  # 中心x座標・上端y座標・面積・実際の楕円ピクセル数
    return None, None, 0

def calc_blue_target_distance(blue_center) -> Optional[int]:
    """
    青ターゲットの位置から走行体が進むべき実際の距離を計算する（非線形計算）。
    
    パラメータ:
        blue_center (tuple or None): find_blue_target_centerから返される(center_x, top_y)
    
    戻り値:
        distance (int or None): 進むべき距離（ピクセル単位での概算距離）、blue_centerがNoneの場合はNone
    
    計算ロジック:
        - カメラサイズ640×480を使用
        - 画像下端90%位置（y=432）を目標位置として設定
        - 非線形補正でY=182→500、Y=330→300程度を実現
        - 実験用パラメータは関数内部で調整可能
    """
    if blue_center is None:
        return None
    
    center_x, top_y = blue_center
    
    # シンプルな線形計算（統一的な制御）
    # カメラサイズ定数を使用して目標位置を設定（480×0.9=432）
    target_y = int(CAMERA_HEIGHT * 0.9)  # 432
    
    # 青ターゲットの上端から目標位置までの距離
    distance = target_y - top_y
    
    # 2点指定線形計算：Y=30で1200、Y=300で430になるよう調整
    if distance > 0:
        # 線形計算：傾き=2.963、切片=18（調整）
        # Y=30(distance=402)→1200、Y=300(distance=132)→410、より低めの距離計算
        slope = 2.963
        intercept = 18  # y=300で410になるよう調整
        practical_distance = int(slope * distance + intercept)
    else:
        # 既に目標位置を通過している場合は短距離
        practical_distance = abs(distance) // 2
    
    # 最小距離を保証（負の値を避ける）
    result = max(practical_distance, 0)
    
    return result

def get_is_blue_line_at_y(image, target_y=470, min_run=30) -> bool:
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
    if image is None or image.size == 0:
        return False
    if not (0 <= target_y < image.shape[0]):
        return False
    # 青色ラインマスク生成（HSV抽出）
    mask = get_color_mask(image, "blue", pattern="line")
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

def get_blue_line_pixel(image) -> int:
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
    if image is None or image.size == 0:
        return 0
    _roi = ROI_LOOP
    x1, y1, x2, y2 = _roi
    # 青色ラインマスク生成（HSV抽出）
    mask_full = get_color_mask(image, "blue", pattern="line")
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

# x=320の中心ラインが赤的（赤い楕円）にヒットしたらTrueを返す関数
def is_x320_on_red_target(image, x_tolerance=60) -> bool:
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
    if image is None or image.size == 0:
        return False
    # 色抽出はget_color_maskで統一
    mask_red = get_color_mask(image, "red", pattern="target")
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
            if area < 300:
                continue
            ellipse = None
            try:
                ellipse = cv2.fitEllipse(cnt)
            except:
                continue
            (cx, cy), (major, minor), angle = ellipse
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            if rect_ratio < 0.4:
                continue
            if area > max_red_area:
                max_red_area = area
                best_center = (int(cx), int(cy))
    if best_center is None:
        return False
    cx, cy = best_center
    if abs(cx - 320) <= x_tolerance:
        return True
    return False

def get_red_target_center_x(image) -> Optional[int]:
    """
    画像内の赤的（楕円）の中心x座標を返す。
    赤的が見つからなければNoneを返す。
    パラメータ:
        img (np.ndarray): BGR画像
    戻り値:
        int or None: 赤的の中心x座標、見つからなければNone
    """
    # 画像がNoneまたは空の場合はNone返却
    if image is None or image.size == 0:
        return None
    # 色抽出はget_color_maskで統一
    mask_red = get_color_mask(image, "red", pattern="target")
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
            if area < 300:
                continue
            ellipse = None
            try:
                ellipse = cv2.fitEllipse(cnt)
            except Exception as e:
                print(f"fitEllipse例外: {e}")
                continue
            (cx, cy), (major, minor), angle = ellipse
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            rect_area = w * h
            rect_ratio = area / rect_area if rect_area > 0 else 0
            if rect_ratio < 0.4:
                continue
            if area > max_red_area:
                max_red_area = area
                best_center = (int(cx), int(cy))
    if best_center is not None:
        return best_center[0]
    return None

# 黒ラインの長さや位置で判定する関数（画像直接渡し、条件はプライベート変数）
def is_left_black_line_detected(image, course) -> bool:
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
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return False
    _min_width = 60
    _min_height = 150
    _min_aspect = 1
    _min_area = 8000
    _roi = ROI_LINE_LEFT
    x1, y1, x2, y2 = _roi
    # leftコース時は左右反転
    if course == 'left':
        image = cv2.flip(image, 1)
    # 緑領域を白で塗りつぶし
    image = fill_green_with_white(image)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
        image,
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

def is_upper_horizontal_line_detected(image) -> bool:
    """
    x=320と交差する一般的な水平黒ラインが検出されたらTrueを返す関数
    ROI: y0から540まで全体、x=320との交差必須、90度に近い角度を重視

    パラメータ:
        img (np.ndarray): BGR画像

    戻り値:
        bool: x=320と交差し90度に近い水平黒ラインが検出されればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return False
    
    # 判定パラメータ
    _min_width = 150      # 幅条件
    _min_height = 10      # 高さ条件
    _max_aspect = 0.2     # アスペクト比（高さ/幅）
    _min_area = 3000      # 面積条件
    _angle_binarize_value = 10  # 角度許容範囲（0度±10または90度±10）
    _center_x = 320       # 画像中心x座標
    _roi = ROI_LINE_HORIZON1
    x1, y1, x2, y2 = _roi

    # 緑領域を白で塗りつぶし
    image = fill_green_with_white(image)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
        image,
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

def is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON2) -> bool:
    """
    x=320を通り、指定されたy座標と交差する水平黒ラインが検出されたらTrueを返す関数。

    Parameters:
        image (np.ndarray): 入力画像（BGR）
        intersection_y (int): 交差判定するy座標（デフォルト: 450）
        roi (tuple): ROI: (x1, y1, x2, y2)（デフォルト: ROI_LINE_HORIZON2）

    Returns:
        bool: 条件を満たす水平黒ラインが検出されればTrue、なければFalse
    """
    # 画像がNoneまたは空の場合はFalse返却
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return False
    
    # ROI座標のみ使用
    x1, y1, x2, y2 = roi

    # 緑領域を白で塗りつぶし
    image = fill_green_with_white(image)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
        image,
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
        crosses_intersection_y = (y <= intersection_y <= y + h)

        # y座標交差かつ面積8000以上でTrue
        if crosses_intersection_y and area >= 5000:
            return True
        
    return False

def is_vertical_black_line_detected(image, roi=ROI_LINE_VERTICAL1, center_tolerance=80) -> bool:
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
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return False
    
    # 判定パラメータ
    _min_width = 50      # 幅条件
    _min_height = 200   # 高さ条件
    _min_aspect = 1.8   # アスペクト比
    _min_area = 11000   # 面積条件
    _center_x = 320     # 画像中心x座標
    x1, y1, x2, y2 = roi

    # 緑領域を白で塗りつぶし
    image = fill_green_with_white(image)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
        image,
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

def get_virtual_line_target_x(image, previous_center_x=None) -> int:
    # 画像がNoneまたは空の場合は320返却
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return 320
    # 仮想ライン検出用ROI座標
    _roi = ROI_VIRTUAL
    x1, y1, x2, y2 = _roi
    mask_full = control_preprocess_image(
        image,
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
            target_x = edge_x + 170
        elif edge_label == 'left':
            edge_x = left_edge_x
            target_x = edge_x - 170
        else:
            # 物体中心には絶対向かわない。安全なデフォルト値。
            target_x = 320

    # previous_center_xによるジャンプ制限（振動抑制）
    if candidates and previous_center_x is not None and 'target_x' in locals():
        max_delta = 10  # 許容する最大変化量を小さく制限
        if abs(target_x - previous_center_x) > max_delta:
            if target_x > previous_center_x:
                target_x = previous_center_x + max_delta
            else:
                target_x = previous_center_x - max_delta

    return target_x

def control_preprocess_image(
    image,
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- グレースケール化 ---
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- コントラスト強調（CLAHE） ---
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=(8,8))
        image = clahe_obj.apply(image)

    # --- フィルター（平滑化） ---
    if blur_type is not None:
        if blur_type == "gaussian":
            image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
        elif blur_type == "median":
            image = cv2.medianBlur(image, blur_ksize)

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
                image = cv2.dilate(image, _MORPHOLOGY_KERNELS['dilate_5x5'], iterations=1)
            elif nr == "dilate7x2":
                image = cv2.dilate(image, _MORPHOLOGY_KERNELS['dilate_7x7'], iterations=2)
            elif nr == "dilate7x3":
                image = cv2.dilate(image, _MORPHOLOGY_KERNELS['dilate_7x7'], iterations=3)
            elif nr == "close3x3":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, _MORPHOLOGY_KERNELS['close_3x3'])
            elif nr == "close5x5_ellipse":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, _MORPHOLOGY_KERNELS['ellipse_5x5'])
            elif nr == "close7x7":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, _MORPHOLOGY_KERNELS['close_7x7'])
            elif nr == "close11x11":
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, _MORPHOLOGY_KERNELS['close_11x11'])
            elif nr == "open3x3":
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, _MORPHOLOGY_KERNELS['open_3x3'])
    return image

def get_color_mask(image, color, pattern=None) -> np.ndarray:
    """
    指定色のHSVマスクを返す（yellow, blue, red対応）。
    patternはbottle/line/targetのみ。未指定時はbottle。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if color == "yellow":
        lower, upper = _HSV_RANGES["yellow"]["bottle"]
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "blue":
        # patternのデフォルト処理（既存の動作と完全一致）
        if pattern == "line":
            lower, upper = _HSV_RANGES["blue"]["line"]
        elif pattern == "target":
            lower, upper = _HSV_RANGES["blue"]["target"]
        else: # bottle or 未指定
            lower, upper = _HSV_RANGES["blue"]["bottle"]
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "red":
        if pattern == "target":
            ranges = _HSV_RANGES["red"]["target"]
        else: # bottle or 未指定
            ranges = _HSV_RANGES["red"]["bottle"]
        lower1, upper1 = ranges[0]
        lower2, upper2 = ranges[1]
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return mask

# --- 緑領域を白で塗りつぶす独立メソッド ---
def fill_green_with_white(image) -> np.ndarray:
    """
    画像の緑領域（HSV指定）を白で塗りつぶす。
    パラメータ:
        img (np.ndarray): BGR画像
    戻り値:
        np.ndarray: 緑領域が白で塗りつぶされた画像（BGR）
    """
    if image.ndim == 3 and image.shape[2] == 3:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green, upper_green = _GREEN_RANGE
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        image[green_mask != 0] = [255, 255, 255]
    return image


# ピンク領域（HSV指定）を白で塗りつぶす関数
def fill_pink_with_white(image) -> np.ndarray:
    """
    画像のピンク領域（HSV指定）を白で塗りつぶす。
    パラメータ:
        image (np.ndarray): BGR画像
    戻り値:
        np.ndarray: ピンク領域が白で塗りつぶされた画像（BGR）
    """
    if image.ndim == 3 and image.shape[2] == 3:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_pink, upper_pink = _PINK_RANGE
        pink_mask = cv2.inRange(hsv_img, lower_pink, upper_pink)
        image[pink_mask != 0] = [255, 255, 255]
    return image

def is_fast_corner_detected(image, roi=ROI_LINE_CORNER, course='right') -> bool:
    """
    ROI内で条件を満たす横棒状物体が検出されたらTrueを返す。
    is_left_black_line_detectedと同じ構造。
    違い: ROI範囲・fill_pink_with_white適用・_target_area=4000・横棒（幅>高さ）条件。
    """
    # 画像がNoneまたは空の場合はFalse返却
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return False
    # 判定条件（関数内定数と同じ値を明示）
    _min_width = 200
    _min_height = 30
    _max_aspect = 1
    _target_area = 3500
    x1, y1, x2, y2 = roi
    # leftコース時は左右反転
    if course == 'left':
        image = cv2.flip(image, 1)
    # 緑・ピンク領域を白で塗りつぶし
    image = fill_green_with_white(image)
    image = fill_pink_with_white(image)
    # 前処理（グレースケール化・CLAHE・メディアンブラー・二値化・ノイズ除去）
    mask_full = control_preprocess_image(
        image,
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
    mask_roi = np.zeros_like(mask_full)
    mask_roi[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = h / (w + 1e-5)
        # 横棒条件: 幅・高さ・アスペクト比・面積で判定
        if w >= _min_width and h >= _min_height and aspect <= _max_aspect and area >= _target_area:
            return True
    return False

# ROI内で中央縦断ラインが条件を満たしているか判定する関数
def is_center_line_detected(img) -> bool:
    """
    ROI内で中央を縦断するラインが、
    ・幅 >= 50
    ・高さ >= 300
    ・面積 >= 10000
    ・ROIの左右端にラインが触れていない
    をすべて満たす場合にTrueを返す。
    画像はBGR想定。
    ROI・閾値等は関数内定数。
    """
    if img is None or img.size == 0:
        return False
    # 内部定数

    _min_width = 50
    _min_height = 300
    _target_area = 10000
    x1, y1, x2, y2 = ROI_LINE_STRAIGHT_FAST
    from .control import fill_green_with_white, control_preprocess_image
    vis_img = fill_green_with_white(img.copy())
    mask_full = control_preprocess_image(
        vis_img,
        use_hsv=False,
        grayscale=True,
        clahe=True,
        clahe_clipLimit=3.0,
        blur_type="median",
        blur_ksize=5,
        binarize_mode="binary_inv",
        binarize_value=120,
        noise_removal=None
    )
    mask_roi = np.zeros_like(mask_full)
    mask_roi[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 元画像絶対座標
        area = cv2.contourArea(cnt)
        cond_w = w >= _min_width
        cond_h = h >= _min_height
        cond_area = area >= _target_area
        cond_left = (x <= x1 + 10)
        cond_right = (x + w >= x2 - 10)
        if cond_w and cond_h and cond_area and (not cond_left) and (not cond_right):
            return True
    return False