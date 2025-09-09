import sys
import os
import traceback

# C++拡張のビルドディレクトリをsys.pathに追加（ラズパイ用想定）
cpp_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../nnspike/utils/c/build'))
if cpp_build_dir not in sys.path:
    sys.path.append(cpp_build_dir)


def test_import_cpp_modules():
    print('sys.version:', sys.version)
    print('sys.path:', sys.path)
    try:
        import control_cpp_get_line_edges_at_y
        print('[OK] import control_cpp_get_line_edges_at_y')
        print('dir(control_cpp_get_line_edges_at_y):', dir(control_cpp_get_line_edges_at_y))
    except Exception as e:
        print('[NG] import control_cpp_get_line_edges_at_y')
        traceback.print_exc()
    try:
        import control_cpp_bottle
        print('[OK] import control_cpp_bottle')
        print('dir(control_cpp_bottle):', dir(control_cpp_bottle))
    except Exception as e:
        print('[NG] import control_cpp_bottle')
        traceback.print_exc()

def test_cpp_functions():
    import numpy as np
    import cv2
    try:
        import control_cpp_get_line_edges_at_y
        # ダミー画像とROIで関数呼び出しテスト
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = (0, 0, 640, 480)
        y = 240
        res = control_cpp_get_line_edges_at_y.get_line_edges_at_y(img, roi, y, 80)
        print('get_line_edges_at_y result:', res)
    except Exception as e:
        print('get_line_edges_at_y failed:')
        traceback.print_exc()
    try:
        import control_cpp_bottle
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        res = control_cpp_bottle.find_bottle_center(img, 'blue', (0, 0, 640, 480))
        print('find_bottle_center result:', res)
    except Exception as e:
        print('find_bottle_center failed:')
        traceback.print_exc()

if __name__ == '__main__':
    print('--- C++拡張importテスト ---')
    test_import_cpp_modules()
    print('\n--- C++関数呼び出しテスト ---')
    test_cpp_functions()
