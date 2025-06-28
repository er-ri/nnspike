import cv2
import numpy as np
from ..constants import ROI_OPENCV, IMAGE_WIDTH, IMAGE_HEIGHT

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
