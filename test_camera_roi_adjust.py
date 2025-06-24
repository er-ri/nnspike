#!/usr/bin/env python3
"""
Camera ROI Adjustment Tool

カメラ画像のROI（Region Of Interest）調整専用プログラム。
指定したROI領域を矩形で表示し、リアルタイムでカメラ画像を確認しながらパラメータ調整が可能。

- 画像ウィンドウ上でROI領域が白枠で表示されます。
- qキーで終了。
- ROI座標はrun_opencv.pyと同じ形式。
"""

import cv2
import os
import datetime
import time

# ROI設定（run_opencv.pyと同じ形式で記述）
ROI_OPENCV = (150, 250, 490, 400)  # (x1, y1, x2, y2)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 動画保存用ディレクトリとファイル名（run_opencv.pyと同じロジック）
TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime())
video_dir = "storage/videos"
os.makedirs(video_dir, exist_ok=True)
video_filename = f"{video_dir}/{TIMESTAMP}_roi_adjust.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter(
    filename=video_filename,
    fourcc=fourcc,
    fps=30,
    frameSize=(IMAGE_WIDTH, IMAGE_HEIGHT),
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

print("カメラ画像ROI調整ツールを起動します")
print(f"ROI_OPENCV: {ROI_OPENCV}")
print("qキーで終了 / sキーで静止画保存 / 動画は自動保存")

save_dir = "output/camera_roi_snapshots"
os.makedirs(save_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラ画像が取得できません")
        break
    # ROI領域を矩形で描画
    x1, y1, x2, y2 = ROI_OPENCV
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, f"ROI: ({x1},{y1})-({x2},{y2})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow('Camera ROI Adjust', frame)
    # 動画として保存
    video_writer.write(frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        # 静止画保存
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/roi_{now}.png"
        cv2.imwrite(filename, frame)
        print(f"画像を保存しました: {filename}")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"動画を保存しました: {video_filename}")
print("終了しました")
