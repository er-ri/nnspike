"""
ライントレーサー制御モジュール

このモジュールは、カメラベースおよび色センサーベースの手法を使用した
ライントレーシングアルゴリズムの実装を含んでいます。これらの実装は
訓練データの収集とリアルタイムロボット制御に使用されます。カメラ画像や
センサー状態などの入力データを送信することで、必要な調整値を返します。

関数:

    calculate_attitude_angle(offset_pixels: float, roi_bottom_y: int,
                           camera_height: float, focal_length_pixels: float) -> float:
        カメラジオメトリを使用してピクセルオフセットから姿勢角（シータ）を計算し、
        より正確なステアリング制御を行います。
"""

import cv2
import math
import numpy as np


def get_line_edges_at_y(image, roi, target_y, threshold_value=50):
    """
    特定のY座標における黒いラインの左右エッジポイントを取得します。

    パラメータ:
    - image: 入力画像（BGRまたはグレースケール）
    - roi_coords: ROIを定義するタプル（x, y, width, height）
    - target_y: ラインエッジを検出するY座標（元画像座標系）
    - threshold_value: 二値変換のしきい値（デフォルト: 50）

    戻り値:
    - left_x: 左エッジのX座標（見つからない場合はNone）
    - right_x: 右エッジのX座標（見つからない場合はNone）
    - line_width: このY位置でのラインの幅（見つからない場合はNone）
    """

    # ROI座標を抽出
    x, y, w, h = roi

    # target_yがROI内にあるかチェック
    if target_y < y or target_y >= y + h:
        return None, None, None

    # 必要に応じてグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # ROIを抽出
    roi = gray[y : y + h, x : x + w]

    # ノイズを減らすためにガウシアンブラーを適用
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # 黒いラインを分離するために二値しきい値処理
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # ROI内の行を計算
    roi_row = target_y - y

    # 目標YでのバイナリRowを取得
    if roi_row >= 0 and roi_row < h:
        row_data = binary[roi_row, :]

        # この行内の全ての白ピクセル（ラインピクセル）を検索
        white_pixels = np.where(row_data == 255)[0]

        if len(white_pixels) > 0:
            # 最も左と最も右の白ピクセルを検索
            left_x_roi = white_pixels[0]
            right_x_roi = white_pixels[-1]

            # 元の画像座標に変換して戻す
            left_x = x + left_x_roi
            right_x = x + right_x_roi
            line_width = right_x - left_x + 1

            return left_x, right_x, line_width

    return None, None, None


def get_all_line_edges_at_y(image, roi, target_y, threshold_value=50, max_edges=None):
    """
    特定のY座標で検出されたすべてのラインエッジを取得します。

    パラメータ:
    - image: 入力画像（BGRまたはグレースケール）
    - roi: ROIを定義するタプル（x, y, width, height）
    - target_y: ラインエッジを検出するY座標（元画像座標系）
    - threshold_value: 二値変換のしきい値（デフォルト: 50）
    - max_edges: 返却する最大エッジ数（デフォルト: すべてのエッジでNone）

    戻り値:
    - 検出されたエッジのx軸座標のリスト: [int, int, ...]
      エッジが見つからない場合は空のリストを返します。
    """

    # ROI座標を抽出
    x, y, w, h = roi

    # target_yがROI内にあるかチェック
    if target_y < y or target_y >= y + h:
        return []

    # 必要に応じてグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # ROIを抽出
    roi_img = gray[y : y + h, x : x + w]

    # ノイズを減らすためにガウシアンブラーを適用
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)

    # 黒いラインを分離するために二値しきい値処理
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # ROI内の行を計算
    roi_row = target_y - y

    # 目標YでのバイナリRowを取得
    if roi_row >= 0 and roi_row < h:
        row_data = binary[roi_row, :]

        # この行内の全ての白ピクセル（ラインピクセル）を検索
        white_pixels = np.where(row_data == 255)[0]

        if len(white_pixels) > 0:
            # 白ピクセルの連続セグメントを見つけて、すべてのエッジを収集
            edges = []
            segment_start = white_pixels[0]

            for i in range(1, len(white_pixels)):
                # 連続する白ピクセル間にギャップがあるかチェック
                if white_pixels[i] - white_pixels[i - 1] > 1:
                    # 現在のセグメントの終了
                    segment_end = white_pixels[i - 1]

                    # 元の画像座標に変換して戻す
                    left_edge = x + segment_start
                    right_edge = x + segment_end

                    # 左右両方のエッジを追加
                    edges.extend([left_edge, right_edge])

                    # 新しいセグメントを開始
                    segment_start = white_pixels[i]

            # 最後のセグメントを忘れずに処理
            segment_end = white_pixels[-1]
            left_edge = x + segment_start
            right_edge = x + segment_end

            # 左右両方のエッジを追加
            edges.extend([left_edge, right_edge])

            # 指定された場合は制限を適用
            if max_edges is not None and len(edges) > max_edges:
                edges = edges[:max_edges]

            return edges

    return []


def find_bottle_center(image):
    """
    OpenCVを使用して画像内のボトルの中心座標を検索します。

    この関数は以下の改善により、リアルタイムアプリケーション用に最適化されています:
    - リアルタイム処理のため、ファイルパスではなくnumpy配列入力を受け入れ
    - 様々な照明条件下でのより良いエッジ検出のために適応的しきい値を使用
    - ノイズと誤検出を減らすために輪郭面積フィルタリングを適用
    - ボトルのような形状を確保するためのアスペクト比検証を含む
    - より良いパフォーマンスのために小さなモルフォロジカルカーネルを使用
    - よりクリーンなリアルタイム動作のためにデバッグprint文を削除

    引数:
        image (numpy.ndarray): numpy配列としての入力画像（BGR形式）

    戻り値:
        tuple: ((x, y), size) ここで(x, y)は中心座標、sizeは最大輪郭の面積
               見つからない場合は(None, None)
    """
    # 画像が有効かチェック
    if image is None or image.size == 0:
        print("エラー: 無効な画像データ")
        return None, None  # より良い検出のために異なる色空間に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 方法1: 色ベース検出（特徴的な色のボトル用）
    # ボトルの色範囲を定義（ボトルの色に基づいて調整）
    # ボトル内の赤い液体用
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 赤色用のマスクを作成
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 方法2: ボトルの輪郭用エッジ検出
    # 様々な照明下でのより良いエッジ検出のために適応的しきい値を使用
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    edges = cv2.bitwise_not(edges)  # エッジを白にするために反転

    # 色とエッジ情報を結合
    combined_mask = cv2.bitwise_or(red_mask, edges)

    # マスクをクリーンアップするためにモルフォロジカル演算を適用
    kernel = np.ones((3, 3), np.uint8)  # リアルタイムパフォーマンス用の小さなカーネル
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel
    )  # 輪郭を検索
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    # ノイズを除去するために面積で輪郭をフィルタ（必要に応じて最小面積を調整）
    min_area = 500  # リアルタイムフィルタリング用の最小面積しきい値
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None

    # 最大の輪郭を検索（ボトルと仮定）
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # 最大輪郭のサイズ（面積）を計算
    contour_size = cv2.contourArea(largest_contour)

    # 追加検証: ボトルのような形状を確保するために輪郭のアスペクト比をチェック
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0

    # ボトルは通常、幅よりも高い（アスペクト比 > 1）
    if aspect_ratio < 0.8:  # 必要に応じてしきい値を調整
        return None, None

    # モーメントを使用して中心を計算
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size

    return None, None


def find_bottle_center_with_yellow_count(image):
    """
    OpenCVを使用して画像内のボトルの中心座標と黄色ピクセル数を検索します。

    この関数は以下の改善により、リアルタイムアプリケーション用に最適化されています:
    - リアルタイム処理のため、ファイルパスではなくnumpy配列入力を受け入れ
    - 様々な照明条件下でのより良いエッジ検出のために適応的しきい値を使用
    - ノイズと誤検出を減らすために輪郭面積フィルタリングを適用
    - ボトルのような形状を確保するためのアスペクト比検証を含む
    - より良いパフォーマンスのために小さなモルフォロジカルカーネルを使用
    - よりクリーンなリアルタイム動作のためにデバッグprint文を削除

    引数:
        image (numpy.ndarray): numpy配列としての入力画像（BGR形式）

    戻り値:
        tuple: ((x, y), size, yellow_pixel_count) ここで(x, y)は中心座標、sizeは最大輪郭の面積、
               yellow_pixel_countは検出された黄色ピクセル数
               見つからない場合は(None, None, 0)
    """
    # 画像が有効かチェック
    if image is None or image.size == 0:
        print("エラー: 無効な画像データ")
        return None, None, 0  # より良い検出のために異なる色空間に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 方法1: 色ベース検出（特徴的な色のボトル用）
    # ボトルの色範囲を定義（ボトルの色に基づいて調整）
    # ボトル内の黄色い液体用（camera.pyのdetect_color_bottleと同じ閾値）
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # 黄色用のマスクを作成
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 黄色ピクセル数を計算
    yellow_pixel_count = cv2.countNonZero(yellow_mask)

    # 方法2: ボトルの輪郭用エッジ検出
    # 様々な照明下でのより良いエッジ検出のために適応的しきい値を使用
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    edges = cv2.bitwise_not(edges)  # エッジを白にするために反転

    # 色とエッジ情報を結合
    combined_mask = cv2.bitwise_or(yellow_mask, edges)

    # マスクをクリーンアップするためにモルフォロジカル演算を適用
    kernel = np.ones((3, 3), np.uint8)  # リアルタイムパフォーマンス用の小さなカーネル
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel
    )  # 輪郭を検索
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, yellow_pixel_count

    # ノイズを除去するために面積で輪郭をフィルタ（必要に応じて最小面積を調整）
    min_area = 500  # リアルタイムフィルタリング用の最小面積しきい値
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None, yellow_pixel_count

    # 最大の輪郭を検索（ボトルと仮定）
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # 最大輪郭のサイズ（面積）を計算
    contour_size = cv2.contourArea(largest_contour)

    # 追加検証: ボトルのような形状を確保するために輪郭のアスペクト比をチェック
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0

    # ボトルは通常、幅よりも高い（アスペクト比 > 1）
    if aspect_ratio < 0.8:  # 必要に応じてしきい値を調整
        return None, None, yellow_pixel_count

    # モーメントを使用して中心を計算
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), contour_size, yellow_pixel_count

    return None, None, yellow_pixel_count


def find_bottle_center_with_blue_count(image):
    """
    OpenCVを使用して画像内のボトルの中心座標と青色ピクセル数を検索します。

    この関数は以下の改善により、リアルタイムアプリケーション用に最適化されています:
    - リアルタイム処理のため、ファイルパスではなくnumpy配列入力を受け入れ
    - 様々な照明条件下でのより良いエッジ検出のために適応的しきい値を使用
    - ノイズと誤検出を減らすために輪郭面積フィルタリングを適用
    - ボトルのような形状を確保するためのアスペクト比検証を含む
    - より良いパフォーマンスのために小さなモルフォロジカルカーネルを使用
    - よりクリーンなリアルタイム動作のためにデバッグprint文を削除

    引数:
        image (numpy.ndarray): numpy配列としての入力画像（BGR形式）

    戻り値:
        tuple: ((x, y), size, blue_pixel_count) ここで(x, y)は中心座標、sizeは最大輪郭の面積、
               blue_pixel_countは検出された青色ピクセル数
               見つからない場合は(None, None, 0)
    """
    # 画像が有効かチェック
    if image is None or image.size == 0:
        print("エラー: 無効な画像データ")
        return None, None, 0  # より良い検出のために異なる色空間に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 方法1: 色ベース検出（特徴的な色のボトル用）
    # ボトルの色範囲を定義（ボトルの色に基づいて調整）
    # ボトル内の青い液体用（camera.pyのdetect_color_bottleと同じ閾値）
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([130, 255, 255])

    # 青色用のマスクを作成
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 青色ピクセル数を計算
    blue_pixel_count = cv2.countNonZero(blue_mask)

    # 方法2: ボトルの輪郭用エッジ検出
    # 様々な照明下でのより良いエッジ検出のために適応的しきい値を使用
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    edges = cv2.bitwise_not(edges)  # エッジを白にするために反転

    # 色とエッジ情報を結合
    combined_mask = cv2.bitwise_or(blue_mask, edges)

    # マスクをクリーンアップするためにモルフォロジカル演算を適用
    kernel = np.ones((3, 3), np.uint8)  # リアルタイムパフォーマンス用の小さなカーネル
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel
    )  # 輪郭を検索
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, blue_pixel_count

    # ノイズを除去するために面積で輪郭をフィルタ（必要に応じて最小面積を調整）
    min_area = 500  # リアルタイムフィルタリング用の最小面積しきい値
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return None, None, blue_pixel_count

    # 最大の輪郭を検索（ボトルと仮定）
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # 最大輪郭のサイズ（面積）を計算
    contour_size = cv2.contourArea(largest_contour)

    # 追加検証: ボトルのような形状を確保するために輪郭のアスペクト比をチェック
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = h / w if w > 0 else 0

    # ボトルは通常、幅よりも高い（アスペクト比 > 1）
    if aspect_ratio < 0.8:  # 必要に応じてしきい値を調整
        return None, None, blue_pixel_count

    # モーメントを使用して中心を計算
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
    カメラジオメトリを使用してピクセルオフセットから姿勢角（シータ）を計算します。

    この関数は、カメラ画像で検出されたピクセルベースのオフセットを、
    ロボットの所望パスからの偏差を表す実世界の姿勢角に変換します。
    これは、単純なピクセルベースの正規化と比較して、より物理的に意味のある
    制御を提供します。

    引数:
        offset_pixels (float): 画像中心からの横方向オフセット（ピクセル）
        roi_bottom_y (int): ROIの下端y座標（ロボットに近い側）
        camera_height (float, optional): 地面からのカメラ高さ（メートル）。デフォルト: 0.20
        focal_length_pixels (float, optional): カメラ焦点距離（ピクセル）。デフォルト: 640

    戻り値:
        float: 姿勢角（シータ）（ラジアン）。正の値は右方向への偏差、
               負の値は左方向への偏差を示します。

    注意:
        正確な角度計算を確保するため、カメラパラメータ（高さと焦点距離）は
        特定のロボットセットアップに対してキャリブレーションされるべきです。
    """
    # カメラからライン検出ポイントまでの地上距離を計算
    # 相似三角形を使用: ground_distance / camera_height = focal_length / (image_height - roi_bottom_y)
    image_height = 480  # 標準カメラ解像度を仮定
    ground_distance = (
        camera_height * focal_length_pixels / (image_height - roi_bottom_y)
    )

    # 横方向オフセットをメートルで計算
    # 相似三角形を使用: lateral_offset / ground_distance = offset_pixels / focal_length
    lateral_offset_meters = offset_pixels * ground_distance / focal_length_pixels

    # アークタンジェントを使用して姿勢角（シータ）を計算
    theta = math.atan2(lateral_offset_meters, ground_distance)

    return theta


def find_gate_center(image):
    """
    Extended ROIを使用してVP13パイプ長方形ゲートの中心座標を検出します。
    
    この関数は以下の特徴を持ちます:
    - Extended ROI: 画面最上部（Y=0）から検索範囲を設定
    - HSV色空間での黒い物体検出により高精度な支柱認識
    - 2つの支柱（左右の足）の検出と中心点計算
    - 面積とアスペクト比による厳密なフィルタリング
    - ゲート特有の形状パターンマッチング
    
    引数:
        image (numpy.ndarray): numpy配列としての入力画像（BGR形式）
    
    戻り値:
        tuple: ((x, y), confidence) ここで(x, y)は中心座標、confidenceは信頼度
               見つからない場合は(None, None)
    """
    # 画像が有効かチェック
    if image is None or image.size == 0:
        print("エラー: 無効な画像データ")
        return None, None
    
    h, w = image.shape[:2]
    
    # Extended ROI設定：最上部まで拡張
    roi_y_start = 0              # 最上部から開始
    roi_y_end = int(h * 0.95)    # 下部95%まで
    roi_x_start = int(w * 0.05)  # 左端5%から
    roi_x_end = int(w * 0.95)    # 右端95%まで
    
    # ROI領域を抽出
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # HSV変換で黒い物体（ゲート支柱）を検出
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 黒い物体の範囲（ゲート支柱用に調整）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])  # 暗い範囲
    black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
    
    # モルフォロジー処理でノイズ除去
    kernel_close = np.ones((3, 3), np.uint8)
    kernel_open = np.ones((2, 2), np.uint8)
    
    black_mask_processed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close)
    black_mask_processed = cv2.morphologyEx(black_mask_processed, cv2.MORPH_OPEN, kernel_open)
    
    # 輪郭検出
    contours, _ = cv2.findContours(black_mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # ゲート支柱候補をフィルタリング
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # 最小面積フィルタ
            continue
            
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        if w_rect < 10 or h_rect < 20:  # 最小サイズフィルタ
            continue
        
        # アスペクト比計算
        aspect_ratio = h_rect / w_rect if w_rect > 0 else 0
        
        # 元の画像座標に変換
        global_center_x = x + w_rect // 2 + roi_x_start
        global_center_y = y + h_rect // 2 + roi_y_start
        
        candidates.append({
            'center': (global_center_x, global_center_y),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'width': w_rect,
            'height': h_rect
        })
    
    # ゲート特有の特徴を持つ候補を選択
    # 目標: 面積1000-1400かつアスペクト比1.0-1.5 または 面積900-1300かつアスペクト比2.0-3.0
    target_candidates = []
    for candidate in candidates:
        area = candidate['area']
        aspect = candidate['aspect_ratio']
        
        if (1000 <= area <= 1400 and 1.0 <= aspect <= 1.5) or \
           (900 <= area <= 1300 and 2.0 <= aspect <= 3.0):
            target_candidates.append(candidate)
    
    # 2つの支柱が見つかった場合
    if len(target_candidates) >= 2:
        # X座標でソート（左から右へ）
        target_candidates.sort(key=lambda c: c['center'][0])
        
        left_leg = target_candidates[0]
        right_leg = target_candidates[1]
        
        # 中心点計算
        center_x = (left_leg['center'][0] + right_leg['center'][0]) // 2
        center_y = (left_leg['center'][1] + right_leg['center'][1]) // 2
        
        # 信頼度計算（面積とアスペクト比の一致度）
        area_score = min(left_leg['area'] / 1000, right_leg['area'] / 1000, 1.0)
        aspect_score = min(left_leg['aspect_ratio'], right_leg['aspect_ratio']) / 3.0
        confidence = (area_score + aspect_score) / 2
        
        return (center_x, center_y), confidence
    
    # フォールバック: 面積が最大の2つの候補を使用
    elif len(candidates) >= 2:
        candidates.sort(key=lambda c: c['area'], reverse=True)
        candidates.sort(key=lambda c: c['center'][0])  # X座標でソート
        
        if len(candidates) >= 2:
            left_leg = candidates[0]
            right_leg = candidates[1]
            
            center_x = (left_leg['center'][0] + right_leg['center'][0]) // 2
            center_y = (left_leg['center'][1] + right_leg['center'][1]) // 2
            
            return (center_x, center_y), 0.5  # 低い信頼度
    
    return None, None
