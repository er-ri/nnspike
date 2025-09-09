


import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))


import cv2
import numpy as np
from nnspike.utils import control

# --- C++拡張のimportパスを絶対パスで追加（control_c.pyと同じロジック） ---
cpp_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../nnspike/utils/c/build'))
cpp_release_dir = os.path.join(cpp_build_dir, 'Release')
for p in [cpp_build_dir, cpp_release_dir]:
    if p not in sys.path:
        sys.path.append(p)

# --- C++拡張を直接import ---
try:
    import control_cpp_get_line_edges_at_y
except ImportError:
    control_cpp_get_line_edges_at_y = None
try:
    import control_cpp_bottle
except ImportError:
    control_cpp_bottle = None



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
    if control_cpp_get_line_edges_at_y is not None:
        res_c = control_cpp_get_line_edges_at_y.get_line_edges_at_y(img, roi, y, th)
    else:
        res_c = None
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
    if control_cpp_get_line_edges_at_y is not None:
        for _ in range(N):
            t2 = time.perf_counter()
            control_cpp_get_line_edges_at_y.get_line_edges_at_y(img, roi, y, th)
            t3 = time.perf_counter()
            c_times.append((t3 - t2) * 1000)  # ms
        c_time = sum(c_times) / N / 1000
    else:
        c_time = 0
    py_time = sum(py_times) / N / 1000
    print(f"Python実装(平均): {py_time*1000:.3f} ms")
    if c_time > 0:
        print(f"C++実装(平均):    {c_time*1000:.3f} ms")
        print(f"速度比 (Python/C++): {py_time/c_time:.2f}倍")
        print(f"C++実装(生):    {[f'{t:.3f}' for t in c_times]}")
    else:
        print("C++拡張がimportできませんでした")
    print(f"Python実装(生): {[f'{t:.3f}' for t in py_times]}")

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


def benchmark_find_bottle_center_python():
    import cv2
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

if __name__ == "__main__":
    benchmark_find_bottle_center_python()
