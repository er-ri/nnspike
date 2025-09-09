import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import time
import cv2
import numpy as np
from nnspike.utils import control



def make_diagonal_line_image(width=640, height=480, line_thickness=10, angle_deg=45, color=(0,0,0)):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白背景
    center = (width // 2, height // 2)
    length = int(np.hypot(width, height))
    angle_rad = np.deg2rad(angle_deg)
    dx = int(np.cos(angle_rad) * length // 2)
    dy = int(np.sin(angle_rad) * length // 2)
    pt1 = (center[0] - dx, center[1] - dy)
    pt2 = (center[0] + dx, center[1] + dy)
    cv2.line(img, pt1, pt2, color, line_thickness)
    return img

def make_curve_image(width=640, height=480, thickness=10, color=(0,0,0)):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    pts = np.array([
        [width//4, height//4],
        [width//2, height//2],
        [3*width//4, 3*height//4]
    ], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], False, color, thickness)
    img4 = make_diagonal_line_image(angle_deg=45, line_thickness=20)
    img5 = make_diagonal_line_image(angle_deg=45, line_thickness=10, color=(0,0,255))
    img6 = make_curve_image(thickness=10)
        return img


def benchmark_find_bottle_center_python():
    img_path = os.path.join(os.path.dirname(__file__), 'frame_76.png')
    img = cv2.imread(img_path)
    if img is None:
        print(f"画像が読み込めません: {img_path}")
        return
    from nnspike.constants import ROI_COLOR
    roi = ROI_COLOR
    color = 'yellow'  # 必要に応じて変更
    print(f"--- find_bottle_center(Python) {img_path}, color={color} ---")
    # 1回ずつの処理時間も計測
    t0 = time.perf_counter()
    res_py = control.find_bottle_center(img, color, roi)
    t1 = time.perf_counter()
    py_once = (t1 - t0) * 1000
    print("Python結果:", res_py)
    print(f"Python実装(1回): {py_once:.3f} ms")
    # N回平均も計測
    N = 100
    py_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        control.find_bottle_center(img, color, roi)
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * 1000)
    py_time = sum(py_times) / N / 1000
    print(f"Python実装(平均): {py_time*1000:.3f} ms")
    print(f"Python実装(生): {[f'{t:.3f}' for t in py_times]}")

def benchmark_get_line_edges_at_y_python():
    roi = (0, 0, 640, 480)
    y = 240
    th = 80
    test_images = [
        (make_diagonal_line_image(angle_deg=0, line_thickness=10), "横直線(10px)"),
        (make_diagonal_line_image(angle_deg=90, line_thickness=10), "縦直線(10px)"),
        (make_diagonal_line_image(angle_deg=45, line_thickness=10), "斜め45度直線(10px)"),
        (make_diagonal_line_image(angle_deg=45, line_thickness=20), "斜め45度直線(20px)"),
        (make_diagonal_line_image(angle_deg=45, line_thickness=10, color=(0,0,255)), "斜め45度赤直線(10px)"),
        (make_curve_image(thickness=10), "曲線(10px)")
    ]
    for img, label in test_images:
        print(f"--- get_line_edges_at_y(Python) [{label}] ---")
        t0 = time.perf_counter()
        res = control.get_line_edges_at_y(img, roi, y, th)
        t1 = time.perf_counter()
        once = (t1 - t0) * 1000
        print("Python結果:", res)
        print(f"Python実装(1回): {once:.3f} ms")
        N = 100
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            control.get_line_edges_at_y(img, roi, y, th)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        avg = sum(times) / N / 1000
        print(f"Python実装(平均): {avg*1000:.3f} ms")
        print(f"Python実装(生): {[f'{t:.3f}' for t in times]}")

def benchmark_get_is_blue_line_at_y_python():
    # ランダム画像でベンチマーク
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    y = 470
    min_run = 30
    print("--- get_is_blue_line_at_y(Python) ---")
    t0 = time.perf_counter()
    res = control.get_is_blue_line_at_y(img, y, min_run)
    t1 = time.perf_counter()
    once = (t1 - t0) * 1000
    print("Python結果:", res)
    print(f"Python実装(1回): {once:.3f} ms")
    N = 100
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        control.get_is_blue_line_at_y(img, y, min_run)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg = sum(times) / N / 1000
    print(f"Python実装(平均): {avg*1000:.3f} ms")
    print(f"Python実装(生): {[f'{t:.3f}' for t in times]}")

if __name__ == "__main__":
    benchmark_find_bottle_center_python()
    benchmark_get_line_edges_at_y_python()
    benchmark_get_is_blue_line_at_y_python()
