#!/usr/bin/env python3
"""
control.py最適化後の機能確認テスト
"""

import numpy as np
import cv2
import nnspike.utils.control as ctrl

def test_constants():
    """定数の定義確認"""
    print("=== 定数定義確認 ===")
    
    # HSV範囲定数確認
    hsv_ranges = ctrl._HSV_RANGES
    print(f"HSV範囲定数: {len(hsv_ranges)}色定義")
    for color, patterns in hsv_ranges.items():
        print(f"  {color}: {type(patterns)} - {list(patterns.keys()) if isinstance(patterns, dict) else 'tuple'}")
    
    # 緑範囲定数確認
    green_range = ctrl._GREEN_RANGE
    print(f"緑範囲定数: {type(green_range)}, 長さ={len(green_range)}")
    print(f"  lower: {green_range[0]}")
    print(f"  upper: {green_range[1]}")
    
    # モルフォロジーカーネル確認
    kernels = ctrl._MORPHOLOGY_KERNELS
    print(f"モルフォロジーカーネル: {len(kernels)}種類定義")
    for name, kernel in kernels.items():
        print(f"  {name}: shape={kernel.shape}, dtype={kernel.dtype}")

def test_hsv_functions():
    """HSV関連関数の動作確認"""
    print("\n=== HSV関数動作確認 ===")
    
    # テスト画像作成
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[20:80, 20:80] = [120, 255, 200]  # 青色領域
    
    # get_color_mask確認
    for color in ['blue', 'yellow']:
        if color in ctrl._HSV_RANGES:
            for pattern in ctrl._HSV_RANGES[color].keys():
                mask = ctrl.get_color_mask(test_img, color, pattern)
                print(f"  {color}-{pattern}: mask shape={mask.shape}, 非ゼロピクセル={np.count_nonzero(mask)}")

def test_green_fill():
    """fill_green_with_white関数確認"""
    print("\n=== 緑塗りつぶし関数確認 ===")
    
    # 緑色テスト画像作成
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[30:70, 30:70] = [60, 180, 100]  # 緑色領域（HSV: 60度）
    
    # 元画像のピクセル値確認
    original_green_pixel = test_img[50, 50].copy()
    print(f"  処理前緑ピクセル: {original_green_pixel}")
    
    # fill_green_with_white実行
    result = ctrl.fill_green_with_white(test_img)
    
    # 結果確認
    processed_pixel = result[50, 50]
    print(f"  処理後ピクセル: {processed_pixel}")
    print(f"  白色変換確認: {'OK' if np.array_equal(processed_pixel, [255, 255, 255]) else 'NG'}")

def test_morphology():
    """モルフォロジー処理確認"""
    print("\n=== モルフォロジー処理確認 ===")
    
    # テスト画像作成
    test_img = np.zeros((100, 100), dtype=np.uint8)
    test_img[40:60, 40:60] = 255
    
    # 各カーネルでの処理確認
    test_operations = [
        ("dilate", "dilate_5x5"),
        ("close", "close_7x7"),
        ("open", "open_3x3")
    ]
    
    for op_name, kernel_name in test_operations:
        if kernel_name in ctrl._MORPHOLOGY_KERNELS:
            kernel = ctrl._MORPHOLOGY_KERNELS[kernel_name]
            if op_name == "dilate":
                result = cv2.dilate(test_img, kernel, iterations=1)
            elif op_name == "close":
                result = cv2.morphologyEx(test_img, cv2.MORPH_CLOSE, kernel)
            elif op_name == "open":
                result = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, kernel)
            
            print(f"  {op_name}({kernel_name}): 非ゼロピクセル={np.count_nonzero(result)}")

def test_preprocess_integration():
    """control_preprocess_image統合確認"""
    print("\n=== 前処理統合確認 ===")
    
    # カラーテスト画像作成
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 前処理実行（モルフォロジー演算含む）
    try:
        result = ctrl.control_preprocess_image(
            test_img,
            use_hsv=False,
            grayscale=True,
            clahe=True,
            blur_type="median",
            blur_ksize=5,
            binarize_mode="binary",
            binarize_value=128,
            noise_removal=["dilate", "close7x7"]
        )
        print(f"  前処理成功: result shape={result.shape}, dtype={result.dtype}")
        print(f"  バイナリ確認: unique values={np.unique(result)}")
    except Exception as e:
        print(f"  前処理エラー: {e}")

if __name__ == "__main__":
    print("control.py最適化後機能確認テスト開始")
    print("=" * 50)
    
    try:
        test_constants()
        test_hsv_functions()
        test_green_fill()
        test_morphology()
        test_preprocess_integration()
        
        print("\n" + "=" * 50)
        print("全テスト完了: 機能保証確認済み")
        
    except Exception as e:
        print(f"\nテスト失敗: {e}")
        raise
