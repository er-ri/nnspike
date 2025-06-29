import cv2
import numpy as np
# from nnspike.constants import ROI_OPENCV, IMAGE_WIDTH, IMAGE_HEIGHT

ROI_OPENCV = (180, 300, 460, 400)  # 左右をそれぞれ30pxずつ内側に狭めた例
ROI_BOTTLE = (130, 50, 510, 400)  # 上部をさらに80px上に拡張（y1=300→220）
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480     

class Camera:
    """
    カメラ操作をカプセル化するクラス。
    cap.read() などのOpenCVカメラ操作を分離し、mainから直接触らない設計。
    画像取得と画像処理（steer_by_camera）も一元化。
    """
    def __init__(self, device_index=0, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=30, roi=ROI_OPENCV):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.roi = roi
        self.image_width = width

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def steer_by_camera(self, frame):
        """
        カメラフレームからROI内の輪郭検出を行い、進行方向の判断に必要な情報を辞書で返す。
        """
        x1, y1, x2, y2 = self.roi
        roi_area = frame[y1:y2, x1:x2]
        image = cv2.cvtColor(roi_area, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            mu = cv2.moments(max_contour)
            mx = mu["m10"] / (mu["m00"] + 1e-5)
            my = mu["m01"] / (mu["m00"] + 1e-5)
        else:
            mx = image.shape[1] / 2
            my = image.shape[0] / 2
            max_contour = None
        roi_center_x = image.shape[1] / 2
        offset_pixels = mx - roi_center_x
        return {
            "mx": mx,
            "my": my,
            "offset_pixels": offset_pixels,
            "max_contour": max_contour
        }

    def detect_bottle(self, frame, roi=ROI_BOTTLE):
        """
        画像内からペットボトルらしい輪郭を検出する。
        - roi: (x1, y1, x2, y2) 独自のROIを指定可能。Noneならデフォルトself.roi。
        戻り値: (bottle_found: bool, bottle_contour: np.ndarray or None)
        """
        x1, y1, x2, y2 = roi
        roi_img = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_img = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / w if w > 0 else 0
            # 条件を緩めに: 面積・アスペクト比・bbox
            if (110000 < area < 130000 and
                1.2 < aspect < 1.7 and
                0 <= x <= 10 and 0 <= y <= 10 and
                250 <= w <= 300 and 350 <= h <= 420):
                return True, cnt
        return False, None

    def detect_color_bottle(self, frame, roi=ROI_BOTTLE):
        """
        ROI内の黄色・青・赤領域の面積が各色ごとのしきい値ピクセル以上ならTrueを返す。
        優先順位: 黄色→青→赤
        Args:
            frame: BGR画像(numpy array)
            roi: (x1, y1, x2, y2)のタプル
        Returns:
            dict: {'result': bool, 'color': str or None, 'pixels': int}
        """
        THRESHOLD_YELLOW = 12000
        THRESHOLD_BLUE = 12000
        THRESHOLD_RED = 12000
        x1, y1, x2, y2 = roi
        roi_img = frame[y1:y2, x1:x2]
        if roi_img is None or roi_img.size == 0:
            return {'result': False, 'color': None, 'pixels': 0}
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        # 赤（2つの範囲）
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
        # 判定（優先順位: 黄→青→赤）
        if yellow_pixels >= THRESHOLD_YELLOW:
            return {'result': True, 'color': 'yellow', 'pixels': yellow_pixels}
        elif blue_pixels >= THRESHOLD_BLUE:
            return {'result': True, 'color': 'blue', 'pixels': blue_pixels}
        elif red_pixels >= THRESHOLD_RED:
            return {'result': True, 'color': 'red', 'pixels': red_pixels}
        else:
            return {'result': False, 'color': None, 'pixels': 0}
