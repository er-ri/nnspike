import cv2
import numpy as np
import threading
import time
from nnspike.constants import CAMERA_WIDTH, CAMERA_HEIGHT

class Video:
    def __init__(self):
        """
        常にrealtimeモードのみ。fps/buffer_sizeは内部定数で管理
        """
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.cap = cv2.VideoCapture(0)  # USBカメラ前提で0固定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.__prev_frame = None
        self.__prev_camera_update = None
        self.__last_update_time = None
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self._frame_id = 0

    def warmup(self, count=10):
        """
        カメラウォームアップ用: 指定回数read()を呼ぶ
        """
        for _ in range(count):
            self.read()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.__last_update_time = time.time()
                if ret and frame is not None:
                    self._frame_id += 1

    def read(self):
        """
        最新フレームのみ返す（スレッドで取得した最新フレーム）
        取得失敗時は None を返す
        """
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame
            else:
                return False, None

    def get_frame(self):
        """
        最新フレームを返す。取得失敗時は前回フレーム→ダミー画像。
        デバッグ出力（更新判定・dt計算・警告）もこの中で行う。
        """
        with self.lock:
            last_update = self.__last_update_time
            frame = self.frame
            ret = self.ret
            updated = False
            camera_dt = None
            if self.__prev_frame is not None and frame is not None:
                updated = (last_update is not None and last_update != self.__prev_camera_update)
            if self.__prev_camera_update is not None and last_update is not None and last_update != self.__prev_camera_update:
                camera_dt = (last_update - self.__prev_camera_update) * 1000
            if camera_dt is not None:
                print(f"[DEBUG] Camera dt={camera_dt:.2f}ms, updated={updated}, id={self._frame_id}")
            else:
                print(f"[DEBUG] Camera not updated, updated={updated}, id={self._frame_id}")
            self.__prev_camera_update = last_update
            if not ret or frame is None:
                if self.__prev_frame is not None:
                    print("[WARN] Camera frame not received. Using previous frame.")
                    frame = self.__prev_frame
                else:
                    print("[WARN] Camera frame not received. Using blank image.")
                    frame = self.dummy_frame
            else:
                self.__prev_frame = frame
            return frame

    def release(self):
        self.running = False
        self.thread.join()
        # self.cap.release()
