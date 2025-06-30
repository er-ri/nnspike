import cv2
import numpy as np
# from nnspike.constants import ROI_OPENCV, IMAGE_WIDTH, IMAGE_HEIGHT

BLACK_THRESHOLD = 130  # 黒判定のしきい値（固定, steer_by_camera用）
ROI_OPENCV = (180, 300, 460, 400)  # 左右をそれぞれ30pxずつ内側に狭めた例
ROI_BOTTLE = (100, 20, 540, 400)  # 上下左右を拡張
IMAGE_WIDTH = 640                  # カメラ画像の幅
IMAGE_HEIGHT = 480     
COLOR_DETECT_PIXEL_THRESHOLD = 3000  # 色領域検出のピクセル数しきい値

class Camera:
    """
    カメラ操作をカプセル化するクラス。
    cap.read() などのOpenCVカメラ操作を分離し、mainから直接触らない設計。
    画像取得と画像処理（steer_by_camera）も一元化。
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

    def steer_by_camera(self, frame):
        """
        ROI_BOTTLE内の色領域（黄→青→赤）の優先順位で進行方向を決定。
        どれもなければ従来通りROI_OPENCV内の黒領域で進行方向を決定。
        戻り値: {'mx': float, 'my': float, 'offset_pixels': float, 'max_contour': contour or None, 'roi_type': str}
        """
        # --- カラー判定を有効化 ---
        x1, y1, x2, y2 = self.roi_bottle
        roi_area = frame[y1:y2, x1:x2]
        color_pixels, color_masks = self.detect_color_bottle(frame)
        for color in ["yellow", "blue", "red"]:
            if color_pixels[color] >= COLOR_DETECT_PIXEL_THRESHOLD:
                mask = cv2.erode(color_masks[color], None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    max_contour = max(contours, key=cv2.contourArea)
                    mu = cv2.moments(max_contour)
                    mx = mu["m10"] / (mu["m00"] + 1e-5)
                    my = mu["m01"] / (mu["m00"] + 1e-5)
                else:
                    mx = roi_area.shape[1] / 2
                    my = roi_area.shape[0] / 2
                    max_contour = None
                roi_center_x = roi_area.shape[1] / 2
                offset_pixels = mx - roi_center_x
                return {"mx": mx, "my": my, "offset_pixels": offset_pixels, "max_contour": max_contour, "roi_type": "bottle"}
        # --- 黒判定はROI_OPENCV ---
        x1, y1, x2, y2 = self.roi
        roi_area = frame[y1:y2, x1:x2]
        image = cv2.cvtColor(roi_area, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        # _, thresh = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
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
        return {"mx": mx, "my": my, "offset_pixels": offset_pixels, "max_contour": max_contour, "roi_type": "opencv"}

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

    def detect_color_bottle(self, frame):
        """
        self.roi_bottle内の黄色・青・赤領域の面積とマスク画像を返す。
        Returns:
            dict: {'yellow': int, 'blue': int, 'red': int}
            masks: {'yellow': mask, 'blue': mask, 'red': mask}
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

    def steer_by_red(self, frame, roi=None):
        """
        ROI_BOTTLE内の赤色領域のみを対象に最大輪郭の重心(mx, my)と中心からのオフセット(offset_pixels)を計算する。
        戻り値: {'mx': float, 'my': float, 'offset_pixels': float, 'max_contour': contour or None}
        """
        if roi is None:
            x1, y1, x2, y2 = ROI_BOTTLE  # ROI_BOTTLEを常に使う
        else:
            x1, y1, x2, y2 = roi
        roi_area = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_area, cv2.COLOR_BGR2HSV)
        # 赤色マスク（2つの範囲を合成）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        # 輪郭抽出
        mask = cv2.erode(red_mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            mu = cv2.moments(max_contour)
            mx = mu["m10"] / (mu["m00"] + 1e-5)
            my = mu["m01"] / (mu["m00"] + 1e-5)
        else:
            mx = roi_area.shape[1] / 2
            my = roi_area.shape[0] / 2
            max_contour = None
        roi_center_x = roi_area.shape[1] / 2
        offset_pixels = mx - roi_center_x
        return {
            "mx": mx,
            "my": my,
            "offset_pixels": offset_pixels,
            "max_contour": max_contour
        }
