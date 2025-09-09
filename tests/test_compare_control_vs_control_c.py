
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import cv2
import numpy as np
from nnspike.utils import control, control_c

def make_dummy_line_image(width=640, height=480, line_y=240, line_thickness=10):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白背景
    cv2.rectangle(img, (0, line_y), (width-1, line_y+line_thickness-1), (0,0,0), -1)  # 黒ライン
    return img

def test_compare_get_line_edges_at_y():
    img = make_dummy_line_image()
    roi = (0, 0, 640, 480)
    y = 245  # ライン中央付近
    th = 80
    res_py = control.get_line_edges_at_y(img, roi, y, th)
    res_c = control_c.get_line_edges_at_y(img, roi, y, th)
    print("Python:", res_py)
    print("C++:", res_c)
    print("一致:", res_py == res_c)
    # 画像保存（目視確認用）
    cv2.imwrite("tests/dummy_line_image.png", img)

if __name__ == "__main__":
    test_compare_get_line_edges_at_y()
