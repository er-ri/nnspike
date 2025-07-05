import cv2
import numpy as np


BLACK_THRESHOLD = 130  # 黒判定のしきい値（固定, steer_by_camera用）
ROI_OPENCV = (10, 50, 630, 450)  # run_opencv.pyと完全一致（左右さらに+30px拡張）
ROI_BOTTLE = (100, 20, 540, 450)  # run_opencv.pyと完全一致
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480     
COLOR_DETECT_PIXEL_THRESHOLD = 3000  # 色領域検出のピクセル数しきい値

OFFSET_Y = 400  # ライン検出Y座標（カメラ画像基準）

class Camera:
    """
    カメラ操作・画像取得・画像処理（ライン/ボトル検出）を一元管理するクラス。
    - OpenCVカメラ操作（cap.read等）はこのクラス内で完結し、main等から直接触らない設計。
    - ROIや解像度、FPSなども初期化時に指定可能。
    - ラインエッジ検出・色検出など現場調整が多い処理もメソッド化。
    """
    def __init__(self, device_index=0, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=30, roi=None, roi_bottle=None):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # ROIはrun_opencv.pyのグローバル定数と完全同期
        self.roi = ROI_OPENCV if roi is None else roi
        self.roi_bottle = ROI_BOTTLE if roi_bottle is None else roi_bottle
        self.image_width = width

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()



    def get_line_edges_at_y(self, image, threshold_value=50):
        """
        ROI内のOFFSET_Y行で黒または青のラインの左右端を検出し、(left_x, right_x, line_width)を返す。
        - left_x, right_x: 画像全体座標でのライン端点
        - line_width: ライン幅（ピクセル数）
        黒・青どちらかのラインが見つかれば検出し、両方重なっていれば両方を統合して検出。
        ラインが見つからない場合も「直前の点」を絶対に離さず維持し続ける（中央や初期値に戻さない）。
        """
        target_y = OFFSET_Y
        x, y, w, h = self.roi

        if not hasattr(self, '_prev_left_x'):
            self._prev_left_x = None
            self._prev_right_x = None

        if target_y < y or target_y >= y + h:
            # ROI外なら直前の点を維持
            if self._prev_left_x is not None and self._prev_right_x is not None:
                return self._prev_left_x, self._prev_right_x, abs(self._prev_right_x - self._prev_left_x) + 1
            # 初回のみ中央
            center_x = x + w // 2
            self._prev_left_x = center_x
            self._prev_right_x = center_x
            return center_x, center_x, 1

        roi = image[y : y + h, x : x + w]

        # 黒ライン用バイナリ
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_black = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # 青ライン用バイナリ（HSV色空間で青領域を抽出）
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 80, 50])
        upper_blue = np.array([130, 255, 255])
        binary_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # 黒ラインはそのまま、青ラインはノイズ除去（小さい点を無視）
        # 青ラインのノイズ除去: ラベリングして十分な幅のものだけ残す
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_blue)
        min_blue_width = 30  # 青ラインとみなす最小幅（ピクセル）
        filtered_blue = np.zeros_like(binary_blue)
        for i in range(1, num_labels):
            left = stats[i, cv2.CC_STAT_LEFT]
            width = stats[i, cv2.CC_STAT_WIDTH]
            if width >= min_blue_width:
                filtered_blue[labels == i] = 255
        # 黒 or "十分な幅の青" のどちらかのピクセルを1にする
        binary = cv2.bitwise_or(binary_black, filtered_blue)

        roi_row = target_y - y
        if 0 <= roi_row < h:
            row_data = binary[roi_row, :]
            white_pixels = np.where(row_data == 255)[0]
            if len(white_pixels) > 0:
                left_x_roi = white_pixels[0]
                right_x_roi = white_pixels[-1]
                left_x = x + left_x_roi
                right_x = x + right_x_roi
                line_width = right_x - left_x + 1
                # 直前の点を更新
                self._prev_left_x = left_x
                self._prev_right_x = right_x
                return left_x, right_x, line_width
        # ラインが見つからない場合も、直前の点を絶対に離さず維持し続ける（中央や初期値に戻さない）
        if self._prev_left_x is not None and self._prev_right_x is not None:
            return self._prev_left_x, self._prev_right_x, abs(self._prev_right_x - self._prev_left_x) + 1
        # 初回のみ中央
        center_x = x + w // 2
        self._prev_left_x = center_x
        self._prev_right_x = center_x
        return center_x, center_x, 1

    def detect_color_bottle(self, frame):
        """
        ROI_BOTTLE内の黄色・青・赤領域のピクセル数とマスク画像を返す。
        戻り値:
            - dict: {'yellow': int, 'blue': int, 'red': int}  # 各色のピクセル数
            - dict: {'yellow': mask, 'blue': mask, 'red': mask}  # 各色の2値マスク画像
        ボトルが検出されない場合はピクセル数0・マスクNoneを返す。
        
        注意: 青色・黄色検出の閾値は統計分析により最適化されています
        """
        # 新しい高度な検出機能を使用
        return self.detect_color_bottle_advanced(frame)
    
    def detect_color_bottle_advanced(self, frame):
        """
        ROI_BOTTLE内の黄色・青・赤領域を統計分析に基づく最適化された手法で検出。
        すべての色に対してペットボトル形状フィルタリングを適用し、背景ノイズを除去。
        
        戻り値:
            - dict: {'yellow': int, 'blue': int, 'red': int}  # 各色のピクセル数
            - dict: {'yellow': mask, 'blue': mask, 'red': mask}  # 各色の2値マスク画像
        """
        x1, y1, x2, y2 = self.roi_bottle
        roi_img = frame[y1:y2, x1:x2]
        if roi_img is None or roi_img.size == 0:
            return {'yellow': 0, 'blue': 0, 'red': 0}, {'yellow': None, 'blue': None, 'red': None}
        
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # 赤（色検出 + ペットボトル形状フィルタリング）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask_initial = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = self._filter_bottle_contours(red_mask_initial)
        
        # 青（統計分析に基づく最適化された閾値 + ペットボトル形状フィルタリング）
        lower_blue = np.array([82, 55, 118])
        upper_blue = np.array([105, 255, 203])
        blue_mask_initial = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_mask = self._filter_bottle_contours(blue_mask_initial)
        
        # 黄（統計分析に基づく最適化された閾値 + ペットボトル形状フィルタリング）
        lower_yellow = np.array([12, 95, 185])
        upper_yellow = np.array([22, 255, 255])
        yellow_mask_initial = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_mask = self._filter_bottle_contours(yellow_mask_initial)
        
        # 面積計算
        red_pixels = int(cv2.countNonZero(red_mask))
        blue_pixels = int(cv2.countNonZero(blue_mask))
        yellow_pixels = int(cv2.countNonZero(yellow_mask))
        
        return (
            {'yellow': yellow_pixels, 'blue': blue_pixels, 'red': red_pixels},
            {'yellow': yellow_mask, 'blue': blue_mask, 'red': red_mask}
        )
    
    def _filter_bottle_contours(self, mask):
        """
        ペットボトル形状フィルタリングをマスクに適用（ノートブック分析結果に基づく）。
        赤、青、黄色すべてのペットボトル検出に使用。
        - 面積: 200-50000px²
        - アスペクト比: 0.3-3.5
        - 円形度: 0.25以上
        """
        # ペットボトル形状フィルタリングパラメータ（ノートブック分析結果）
        MIN_AREA = 200
        MAX_AREA = 50000
        MIN_ASPECT_RATIO = 0.3
        MAX_ASPECT_RATIO = 3.5
        MIN_CIRCULARITY = 0.25
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # フィルタリング済みマスクを作成
        filtered_mask = np.zeros_like(mask)
        
        for cnt in contours:
            # 面積チェック
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            
            # バウンディングボックスとアスペクト比
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue
            
            # 円形度
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < MIN_CIRCULARITY:
                    continue
            
            # 条件を満たす輪郭をマスクに描画
            cv2.fillPoly(filtered_mask, [cnt], 255)
        
        return filtered_mask
