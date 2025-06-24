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

# ROI設定（run_opencv.pyと同じ形式で記述）
ROI_OPENCV = (150, 250, 490, 400)  # (x1, y1, x2, y2)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

print("カメラ画像ROI調整ツールを起動します")
print(f"ROI_OPENCV: {ROI_OPENCV}")
print("qキーで終了")

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
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("終了しました")
