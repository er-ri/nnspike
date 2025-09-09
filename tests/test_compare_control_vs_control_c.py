
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import cv2
import numpy as np
from nnspike.utils import control, control_c


def make_diagonal_line_image(width=640, height=480, line_thickness=10, angle_deg=45):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白背景
    # 画像中心を通る45度の直線を描画
    center = (width // 2, height // 2)
    length = int(np.hypot(width, height))
    angle_rad = np.deg2rad(angle_deg)
    dx = int(np.cos(angle_rad) * length // 2)
    dy = int(np.sin(angle_rad) * length // 2)
    pt1 = (center[0] - dx, center[1] - dy)
    pt2 = (center[0] + dx, center[1] + dy)
    cv2.line(img, pt1, pt2, (0,0,0), line_thickness)
    return img


def test_compare_get_line_edges_at_y():
    img = make_diagonal_line_image()
    roi = (0, 0, 640, 480)
    y = 245  # ライン中央付近
    th = 80
    res_py = control.get_line_edges_at_y(img, roi, y, th)
    res_c = control_c.get_line_edges_at_y(img, roi, y, th)
    print("Python:", res_py)
    print("C++:", res_c)
    print("一致:", res_py == res_c)
    # 画像保存（目視確認用）
    cv2.imwrite("tests/diagonal_line_image.png", img)

if __name__ == "__main__":
    test_compare_get_line_edges_at_y()
