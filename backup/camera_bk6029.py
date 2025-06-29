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
