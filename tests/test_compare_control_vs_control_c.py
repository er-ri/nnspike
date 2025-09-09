import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import time
import cv2
import numpy as np
from nnspike.utils import control


def make_virtual_line_image(width=640, height=480, line_thickness=10, color=(0,0,0)):
    """ROI_VIRTUAL(100,100,540,330)内に確実に黒線が通る画像を生成"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # ROI_VIRTUAL内に2つの黒い矩形（面積・アスペクト比・最大面積条件を厳密に満たす物体）を描画
    x1, y1, x2, y2 = 100, 100, 540, 330
    # 面積: 2000～8000程度、アスペクト比: 1.5～3.0、ROI内に完全に収まるように設計
    # 左側の矩形（やや縦長、上寄り）
    w1, h1 = 60, 35  # area=2100, aspect=1.71
    rect1 = (x1 + 20, y1 + 15, w1, h1)
    # 右側の矩形（やや横長、下寄り＆右寄り）
    w2, h2 = 90, 30  # area=2700, aspect=3.0
    rect2 = (x2 - w2 - 20, y2 - h2 - 10, w2, h2)
    # ROI内に完全に収まることをassertで保証
    assert x1 <= rect1[0] < rect1[0]+w1 <= x2
    assert y1 <= rect1[1] < rect1[1]+h1 <= y2
    assert x1 <= rect2[0] < rect2[0]+w2 <= x2
    assert y1 <= rect2[1] < rect2[1]+h2 <= y2
    # 描画
    cv2.rectangle(img, (rect1[0], rect1[1]), (rect1[0]+rect1[2], rect1[1]+rect1[3]), color, -1)
    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[0]+rect2[2], rect2[1]+rect2[3]), color, -1)
    return img



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
    # 画像を保存して目視確認
    cv2.imwrite("test_find_bottle_center.png", img)
    # 1回ずつの処理時間も計測
    t0 = time.perf_counter()
    res_py = control.find_bottle_center(img, color, roi)
    t1 = time.perf_counter()
    py_once = (t1 - t0) * 1000
    print(f"find_bottle_center(Python) 結果: {res_py}")
    print(f"find_bottle_center(Python) 実装(1回): {py_once:.3f} ms")
    # N回平均も計測
    N = 10
    py_times = []
    for i in range(N):
        start = time.perf_counter()
        result = control.find_bottle_center(img, color=color)
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        print(f"find_bottle_center(Python)[{i}] 結果: {result}, 時間: {elapsed:.3f} ms")

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
        for i in range(N):
            t0 = time.perf_counter()
            res = control.get_line_edges_at_y(img, roi, y, th)
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000
            print(f"get_line_edges_at_y(Python)[{i}] 結果: {res}, 時間: {elapsed:.3f} ms")

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
    for i in range(N):
        t0 = time.perf_counter()
        res = control.get_is_blue_line_at_y(img, y, min_run)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        print(f"get_is_blue_line_at_y(Python)[{i}] 結果: {res}, 時間: {elapsed:.3f} ms")


def benchmark_get_virtual_line_target_x_python():
    # 直線画像でテスト
    roi = (0, 0, 640, 480)
    y = 240
    th = 80
    img = make_virtual_line_image(line_thickness=10)
    # 画像を保存して目視確認（絶対パス指定でtestsディレクトリ内に保存）
    save_path = os.path.join(os.path.dirname(__file__), "test_virtual_line.png")
    cv2.imwrite(save_path, img)
    print("--- get_virtual_line_target_x(Python) [横直線(10px)] ---")
    t0 = time.perf_counter()
    try:
        # get_virtual_line_target_xの仕様に合わせて引数を修正
        # まずimgのみで呼び出し、必要ならyも渡す
        res = control.get_virtual_line_target_x(img)
    except Exception as e:
        print(f"Error: {e}")
        return
    t1 = time.perf_counter()
    once = (t1 - t0) * 1000
    print("Python結果:", res)
    print(f"Python実装(1回): {once:.3f} ms")
    N = 10
    times = []
    for i in range(N):
        t0 = time.perf_counter()
        res = control.get_virtual_line_target_x(img)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        print(f"get_virtual_line_target_x(Python)[{i}] 結果: {res}, 時間: {elapsed:.3f} ms")

if __name__ == "__main__":
    benchmark_find_bottle_center_python()
    # benchmark_get_line_edges_at_y_python()
    # benchmark_get_is_blue_line_at_y_python()
    benchmark_get_virtual_line_target_x_python()
