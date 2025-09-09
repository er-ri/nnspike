
import sys, os
import time
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
    # Python実装の速度計測
    t0 = time.perf_counter()
    res_py = control.get_line_edges_at_y(img, roi, y, th)
    t1 = time.perf_counter()
    py_time = t1 - t0
    # C++実装の速度計測
    t2 = time.perf_counter()
    res_c = control_c.get_line_edges_at_y(img, roi, y, th)
    t3 = time.perf_counter()
    c_time = t3 - t2
    print("Python:", res_py)
    print("C++:", res_c)
    print("一致:", res_py == res_c)
    print(f"Python実装: {py_time*1000:.3f} ms")
    print(f"C++実装:    {c_time*1000:.3f} ms")
    if c_time > 0:
        print(f"速度比 (Python/C++): {py_time/c_time:.2f}倍")
    # 画像保存（目視確認用）
    cv2.imwrite("tests/diagonal_line_image.png", img)

if __name__ == "__main__":
    test_compare_get_line_edges_at_y()
