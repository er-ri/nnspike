

import sys, os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import cv2
import numpy as np
from nnspike.utils import control, control_c



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
    return img



def run_benchmark(img, roi, y, th, label):
    res_py = control.get_line_edges_at_y(img, roi, y, th)
    res_c = control_c.get_line_edges_at_y(img, roi, y, th)
    print(f"--- {label} ---")
    print("Python:", res_py)
    print("C++:", res_c)
    print("一致:", res_py == res_c)

    N = 100
    py_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        control.get_line_edges_at_y(img, roi, y, th)
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * 1000)  # ms
    c_times = []
    for _ in range(N):
        t2 = time.perf_counter()
        control_c.get_line_edges_at_y(img, roi, y, th)
        t3 = time.perf_counter()
        c_times.append((t3 - t2) * 1000)  # ms
    py_time = sum(py_times) / N / 1000
    c_time = sum(c_times) / N / 1000
    print(f"Python実装(平均): {py_time*1000:.3f} ms")
    print(f"C++実装(平均):    {c_time*1000:.3f} ms")
    if c_time > 0:
        print(f"速度比 (Python/C++): {py_time/c_time:.2f}倍")
    print(f"Python実装(生): {[f'{t:.3f}' for t in py_times]}")
    print(f"C++実装(生):    {[f'{t:.3f}' for t in c_times]}")

def test_compare_get_line_edges_at_y():
    roi = (0, 0, 640, 480)
    y = 245
    th = 80
    # バリエーション: 直線
    img1 = make_diagonal_line_image(angle_deg=0, line_thickness=10)
    run_benchmark(img1, roi, y, th, "横直線(10px)")
    img2 = make_diagonal_line_image(angle_deg=90, line_thickness=10)
    run_benchmark(img2, roi, y, th, "縦直線(10px)")
    img3 = make_diagonal_line_image(angle_deg=45, line_thickness=10)
    run_benchmark(img3, roi, y, th, "斜め45度直線(10px)")
    img4 = make_diagonal_line_image(angle_deg=45, line_thickness=20)
    run_benchmark(img4, roi, y, th, "斜め45度直線(20px)")
    img5 = make_diagonal_line_image(angle_deg=45, line_thickness=10, color=(0,0,255))
    run_benchmark(img5, roi, y, th, "斜め45度赤直線(10px)")
    img6 = make_curve_image(thickness=10)
    run_benchmark(img6, roi, y, th, "曲線(10px)")

def test_compare_find_bottle_center():
    import cv2
    img_path = os.path.join(os.path.dirname(__file__), 'frame_76.png')
    img = cv2.imread(img_path)
    if img is None:
        print(f"画像が読み込めません: {img_path}")
        return
    roi = (0, 0, img.shape[1], img.shape[0])
    color = 'blue'  # 必要に応じて変更
    print(f"--- find_bottle_center({img_path}, color={color}) ---")
    res_py = control.find_bottle_center(img, color, roi)
    res_c = control_c.find_bottle_center(img, color, roi)
    print("Python:", res_py)
    print("C++:", res_c)
    print("一致:", res_py == res_c)
    N = 100
    py_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        control.find_bottle_center(img, color, roi)
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * 1000)
    c_times = []
    for _ in range(N):
        t2 = time.perf_counter()
        control_c.find_bottle_center(img, color, roi)
        t3 = time.perf_counter()
        c_times.append((t3 - t2) * 1000)
    py_time = sum(py_times) / N / 1000
    c_time = sum(c_times) / N / 1000
    print(f"Python実装(平均): {py_time*1000:.3f} ms")
    print(f"C++実装(平均):    {c_time*1000:.3f} ms")
    if c_time > 0:
        print(f"速度比 (Python/C++): {py_time/c_time:.2f}倍")
    print(f"Python実装(生): {[f'{t:.3f}' for t in py_times]}")
    print(f"C++実装(生):    {[f'{t:.3f}' for t in c_times]}")

if __name__ == "__main__":
    #test_compare_get_line_edges_at_y()
    test_compare_find_bottle_center()
