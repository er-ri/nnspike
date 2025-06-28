import cv2
import numpy as np
# from nnspike.constants import ROI_OPENCV, IMAGE_WIDTH, IMAGE_HEIGHT

ROI_OPENCV = (180, 300, 460, 400)  # 左右をそれぞれ30pxずつ内側に狭めた例
ROI_BOTTLE = (180, 0, 460, 400)  # 上部をさらに80px上に拡張（y1=300→220）
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

    def detect_bottle(self, frame, min_area=2000, aspect_min=1.8, aspect_max=4.5, roi=ROI_BOTTLE):
        """
        画像内からペットボトルらしい輪郭を検出する。
        - min_area: 輪郭の最小面積
        - aspect_min, aspect_max: アスペクト比(高さ/幅)の範囲
        - roi: (x1, y1, x2, y2) 独自のROIを指定可能。Noneならデフォルトself.roi。
        戻り値: (bottle_found: bool, bottle_contour: np.ndarray or None)
        """
        # --- 独自ROIを使う場合はそちらを優先 ---
        if roi is None:
            x1, y1, x2, y2 = self.roi
        else:
            x1, y1, x2, y2 = roi
        # 上方向にROIを拡張（例: y1を小さくする）
        y1 = max(0, y1 - 80)  # 80px分上に拡張（必要に応じて調整）
        roi_area = frame[y1:y2, x1:x2]
        image = cv2.cvtColor(roi_area, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / (w + 1e-5)
            # 許容幅を持たせた条件
            if (
                100000 < area < 120000 and
                1.3 < aspect < 1.6 and
                x < 10 and y < 10 and 250 < w < 300 and 380 < h < 420
            ):
                return True, cnt
            # 既存の条件（縦長の輪郭）
            if area < min_area:
                continue
            if aspect_min < aspect < aspect_max:
                # ペットボトルらしい縦長の輪郭
                return True, cnt
        return False, None
