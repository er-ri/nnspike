import cv2
import numpy as np
import time

def benchmark_resize(img, n_iter=100):
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter

def benchmark_blur(img, n_iter=100):
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = cv2.GaussianBlur(img, (5, 5), 0)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter

def main():
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    n_iter = 200
    resize_time = benchmark_resize(img, n_iter)
    blur_time = benchmark_blur(img, n_iter)
    print(f"OpenCVバージョン: {cv2.__version__}")
    print(f"リサイズ平均処理時間: {resize_time*1000:.3f} ms")
    print(f"ガウシアンブラー平均処理時間: {blur_time*1000:.3f} ms")

if __name__ == "__main__":
    main()
