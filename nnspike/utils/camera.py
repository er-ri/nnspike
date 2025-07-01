import cv2
import numpy as np
# from nnspike.constants import ROI_OPENCV, IMAGE_WIDTH, IMAGE_HEIGHT

BLACK_THRESHOLD = 130  # 黒判定のしきい値（固定, steer_by_camera用）
ROI_OPENCV = (20, 50, 620, 400)  # OpenCVのROI領域 # (x, y, width, height)
ROI_BOTTLE = (100, 20, 540, 400)  # 上下左右を拡張
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480     
COLOR_DETECT_PIXEL_THRESHOLD = 3000  # 色領域検出のピクセル数しきい値

OFFSET_Y =350  # ライン検出Y座標（カメラ画像基準）

class Camera:
    """
    カメラ操作・画像取得・画像処理（ライン/ボトル検出）を一元管理するクラス。
    - OpenCVカメラ操作（cap.read等）はこのクラス内で完結し、main等から直接触らない設計。
    - ROIや解像度、FPSなども初期化時に指定可能。
    - ラインエッジ検出・色検出など現場調整が多い処理もメソッド化。
    """
    def __init__(self, device_index=0, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=30, roi=ROI_OPENCV, roi_bottle=ROI_BOTTLE):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.roi = roi
        self.roi_bottle = roi_bottle
        self.image_width = width

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get_line_edges_at_y(self, image, threshold_value=50):
        """
        ROI内のOFFSET_Y行でラインの左右端を検出し、(left_x, right_x, line_width)を返す。
        - left_x, right_x: 画像全体座標でのライン端点
        - line_width: ライン幅（ピクセル数）
        ラインが見つからない場合は (None, None, None) を返す。
        """
        target_y = OFFSET_Y
        # Extract ROI coordinates
        x, y, w, h = self.roi

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
        if 0 <= roi_row < h:
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

    def detect_color_bottle(self, frame):
        """
        ROI_BOTTLE内の黄色・青・赤領域のピクセル数とマスク画像を返す。
        戻り値:
            - dict: {'yellow': int, 'blue': int, 'red': int}  # 各色のピクセル数
            - dict: {'yellow': mask, 'blue': mask, 'red': mask}  # 各色の2値マスク画像
        ボトルが検出されない場合はピクセル数0・マスクNoneを返す。
        """
        x1, y1, x2, y2 = self.roi_bottle
        roi_img = frame[y1:y2, x1:x2]
        if roi_img is None or roi_img.size == 0:
            return {'yellow': 0, 'blue': 0, 'red': 0}, {'yellow': None, 'blue': None, 'red': None}
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        # 赤
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        # 青
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 黄
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # 面積計算
        red_pixels = int(cv2.countNonZero(red_mask))
        blue_pixels = int(cv2.countNonZero(blue_mask))
        yellow_pixels = int(cv2.countNonZero(yellow_mask))
        return (
            {'yellow': yellow_pixels, 'blue': blue_pixels, 'red': red_pixels},
            {'yellow': yellow_mask, 'blue': blue_mask, 'red': red_mask}
        )
