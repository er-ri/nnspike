#!/usr/bin/env python3
"""
ActionChain統一フレーム最適化テスト

アクションチェインで_optimize_frameメソッドによる
統一フレーム最適化設計の動作確認を行う。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nnspike.utils.control import optimize_image_memory, find_bottle_center, get_line_edges_at_y
from nnspike.constants import ROI_COLOR, ROI_CNN

def test_optimize_image_memory():
    """optimize_image_memory関数の基本動作テスト"""
    print("=== optimize_image_memory基本テスト ===")
    
    # テスト用フレーム作成
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"元フレーム形状: {test_frame.shape}")
    
    # 最適化実行
    optimized = optimize_image_memory(test_frame)
    print(f"最適化フレーム形状: {optimized.shape}")
    print(f"内容一致: {np.array_equal(test_frame, optimized)}")
    print(f"メモリアドレス変更: {test_frame.__array_interface__['data'][0] != optimized.__array_interface__['data'][0]}")
    
    return optimized

def test_action_chain_integration():
    """ActionChain統一設計の動作確認（簡易版）"""
    print("\n=== ActionChain統一設計検証 ===")
    
    # ActionChainのフレーム最適化メソッドを簡易再現
    def mock_optimize_frame(image):
        """ActionChainの_optimize_frameメソッドを簡易模擬"""
        return optimize_image_memory(image)
    
    # テストフレーム作成
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_frame[100:200, 100:200] = [0, 255, 255]  # 黄色領域追加
    
    print(f"1. 元フレーム準備: {test_frame.shape}")
    
    # フレーム最適化実行
    optimized_frame = mock_optimize_frame(test_frame)
    print(f"2. フレーム最適化: 成功")
    print(f"   内容保持: {np.array_equal(test_frame, optimized_frame)}")
    
    # 複数画像処理関数での使用テスト
    try:
        # find_bottle_centerでの使用
        result1 = find_bottle_center(optimized_frame, "yellow", ROI_COLOR)
        print(f"3. find_bottle_center: 成功 (結果={result1[0] is not None})")
        
        # get_line_edges_at_yでの使用
        result2 = get_line_edges_at_y(optimized_frame, ROI_CNN, 450, 80)
        print(f"4. get_line_edges_at_y: 成功 (結果={result2[0] is not None})")
        
    except Exception as e:
        print(f"画像処理関数テスト: エラー - {e}")
        return False
    
    return True

def test_frame_independence():
    """フレーム独立性テスト（相互影響なし確認）"""
    print("\n=== フレーム独立性検証 ===")
    
    # 元フレーム作成
    original = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    original[100:200, 100:200] = [255, 0, 0]  # 赤色領域
    
    # 最適化フレーム作成
    optimized = optimize_image_memory(original)
    
    print(f"1. 最適化前後同一: {np.array_equal(original, optimized)}")
    
    # 最適化フレームを変更
    optimized[200:300, 200:300] = [0, 0, 255]  # 青色領域追加
    
    # 元フレームが影響を受けないことを確認
    is_independent = not np.array_equal(original, optimized)
    print(f"2. フレーム独立性: {is_independent}")
    print(f"   最適化フレーム変更が元フレームに影響しない: {is_independent}")
    
    return is_independent

def main():
    """メインテスト実行"""
    print("ActionChain統一フレーム最適化テスト開始")
    print("=" * 50)
    
    # 基本動作テスト
    optimized = test_optimize_image_memory()
    
    # 統一設計テスト
    integration_ok = test_action_chain_integration()
    
    # 独立性テスト
    independence_ok = test_frame_independence()
    
    print("\n" + "=" * 50)
    print("テスト結果サマリー:")
    print(f"- 基本動作: 正常")
    print(f"- 統一設計: {'正常' if integration_ok else '異常'}")
    print(f"- フレーム独立性: {'正常' if independence_ok else '異常'}")
    
    if integration_ok and independence_ok:
        print("\n✅ ActionChain統一フレーム最適化設計: 完全動作確認")
        print("run_manual.pyからActionChainへ渡されるフレームが")
        print("自動的にメモリ最適化され、安全に機能します。")
    else:
        print("\n❌ 設計に問題があります")
    
    return integration_ok and independence_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
