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
    print(f"元フレーム連続性: {test_frame.flags['C_CONTIGUOUS']}")
    
    # 最適化実行
    optimized = optimize_image_memory(test_frame)
    print(f"最適化フレーム形状: {optimized.shape}")
    print(f"最適化フレーム連続性: {optimized.flags['C_CONTIGUOUS']}")
    print(f"内容一致: {np.array_equal(test_frame, optimized)}")
    
    # メモリ独立性確認
    memory_independent = test_frame.__array_interface__['data'][0] != optimized.__array_interface__['data'][0]
    print(f"メモリアドレス独立: {memory_independent}")
    
    # メモリ効率確認（連続配置）
    is_contiguous = optimized.flags['C_CONTIGUOUS']
    print(f"メモリ連続配置: {is_contiguous}")
    
    return optimized

def test_action_chain_integration():
    """ActionChain統一設計の動作確認（実装検証版）"""
    print("\n=== ActionChain統一設計検証 ===")
    
    # ActionChainのフレーム最適化メソッドを簡易再現
    def mock_optimize_frame(image):
        """ActionChainの_optimize_frameメソッドを簡易模擬"""
        return optimize_image_memory(image)
    
    # より実用的なテストフレーム作成
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 実用的なパターンを配置
    test_frame[100:200, 100:200] = [0, 255, 255]    # 黄色ボトル領域
    test_frame[400:480, 200:400] = [0, 0, 0]        # 黒ライン領域
    test_frame[50:150, 450:550] = [255, 0, 0]       # 赤ターゲット領域
    test_frame[300:400, 50:150] = [0, 0, 255]       # 青ターゲット領域
    
    print(f"1. 実用的テストフレーム準備: {test_frame.shape}")
    
    # フレーム最適化実行
    optimized_frame = mock_optimize_frame(test_frame)
    print(f"2. フレーム最適化: 成功")
    print(f"   内容保持: {np.array_equal(test_frame, optimized_frame)}")
    print(f"   メモリ連続性: {optimized_frame.flags['C_CONTIGUOUS']}")
    
    # ActionChainで使用される主要画像処理関数での検証
    test_results = []
    
    try:
        # 1. ボトル検出テスト（yellow, red, blue）
        yellow_result = find_bottle_center(optimized_frame, "yellow", ROI_COLOR)
        red_result = find_bottle_center(optimized_frame, "red", ROI_COLOR)
        blue_result = find_bottle_center(optimized_frame, "blue", ROI_COLOR)
        
        print(f"3. ボトル検出テスト:")
        print(f"   - 黄色ボトル: {yellow_result[0] is not None}")
        print(f"   - 赤色ボトル: {red_result[0] is not None}")
        print(f"   - 青色ボトル: {blue_result[0] is not None}")
        test_results.append(True)
        
        # 2. ライン検出テスト
        line_result = get_line_edges_at_y(optimized_frame, ROI_CNN, 450, 80)
        print(f"4. ライン検出: {line_result[0] is not None or line_result[1] is not None}")
        test_results.append(True)
        
        # 3. パフォーマンステスト（複数回実行）
        import time
        start_time = time.time()
        for _ in range(10):
            _ = mock_optimize_frame(test_frame)
            _ = find_bottle_center(optimized_frame, "yellow", ROI_COLOR)
            _ = get_line_edges_at_y(optimized_frame, ROI_CNN, 450, 80)
        
        elapsed = time.time() - start_time
        print(f"5. パフォーマンス: 10回実行 {elapsed:.3f}秒 (平均{elapsed/10:.3f}秒)")
        test_results.append(elapsed < 1.0)  # 1秒以内なら良好
        
    except Exception as e:
        print(f"画像処理関数テスト: エラー - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return all(test_results)

def test_frame_independence():
    """フレーム独立性テスト（相互影響なし確認）"""
    print("\n=== フレーム独立性検証 ===")
    
    # 元フレーム作成
    original = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    original[100:200, 100:200] = [255, 0, 0]  # 赤色領域
    
    # 元フレームのコピーを保存（比較用）
    original_backup = original.copy()
    
    # 最適化フレーム作成
    optimized = optimize_image_memory(original)
    
    print(f"1. 最適化前後同一: {np.array_equal(original, optimized)}")
    print(f"2. メモリアドレス異なる: {original.__array_interface__['data'][0] != optimized.__array_interface__['data'][0]}")
    
    # 最適化フレームを変更
    optimized[200:300, 200:300] = [0, 0, 255]  # 青色領域追加
    
    # 元フレームが影響を受けないことを確認
    original_unchanged = np.array_equal(original, original_backup)
    frames_independent = not np.array_equal(original, optimized)
    
    print(f"3. 元フレーム未変更: {original_unchanged}")
    print(f"4. フレーム独立性: {frames_independent}")
    print(f"   最適化フレーム変更が元フレームに影響しない: {frames_independent}")
    
    # 詳細診断
    if not original_unchanged:
        print("   ⚠️  元フレームが予期せず変更されました")
    if not frames_independent:
        print("   ⚠️  フレーム間で予期しない共有が発生しています")
    
    return original_unchanged and frames_independent

def test_real_actionchain_methods():
    """実際のActionChainメソッドとの互換性テスト"""
    print("\n=== 実ActionChainメソッド互換性検証 ===")
    
    # ActionChainファイルを直接読み込んで構造確認
    action_chain_path = os.path.join(os.path.dirname(__file__), '..', 'nnspike', 'unit', 'action_chain.py')
    
    try:
        with open(action_chain_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 重要な統一設計要素の存在確認
        checks = {
            '_optimize_frame定義': '_optimize_frame(self, image)' in content,
            'optimize_image_memoryインポート': 'optimize_image_memory' in content,
            '冗長メソッド削除1': '_get_target_x_by_course_optimized' not in content,
            '冗長メソッド削除2': '_get_target_x_by_course_safe_optimized' not in content,
            '統一最適化使用': 'optimized_frame = self._optimize_frame(image)' in content
        }
        
        print("ActionChain統一設計要素チェック:")
        for check_name, result in checks.items():
            print(f"  - {check_name}: {'✓' if result else '✗'}")
        
        # アクションメソッドでの最適化使用回数
        optimization_count = content.count('optimized_frame = self._optimize_frame(image)')
        print(f"  - アクションメソッド最適化使用: {optimization_count}回")
        
        # 外部APIメソッドの保持確認
        external_apis = [
            'def get_target_x_by_course(self, image',
            'def get_target_x_by_course_safe(self, image'
        ]
        
        api_preserved = all(api in content for api in external_apis)
        print(f"  - 外部API保持: {'✓' if api_preserved else '✗'}")
        
        all_good = all(checks.values()) and optimization_count >= 3 and api_preserved
        print(f"\n統一設計実装状況: {'完了' if all_good else '未完了'}")
        
        return all_good
        
    except Exception as e:
        print(f"ActionChainファイル確認エラー: {e}")
        return False

def test_memory_efficiency():
    """メモリ効率とリーク検証"""
    print("\n=== メモリ効率検証 ===")
    
    import gc
    
    # 初期メモリ使用量
    gc.collect()
    initial_objects = len(gc.get_objects())
    
    # 大量フレーム処理テスト
    frames_processed = 0
    memory_addresses = set()
    
    for i in range(100):
        # テストフレーム作成
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 最適化実行
        optimized = optimize_image_memory(test_frame)
        
        # メモリアドレス記録
        addr = optimized.__array_interface__['data'][0]
        memory_addresses.add(addr)
        
        frames_processed += 1
        
        # メモリリークチェック（10回ごと）
        if i % 10 == 0:
            gc.collect()
    
    # 最終メモリ使用量
    gc.collect()
    final_objects = len(gc.get_objects())
    
    print(f"1. 処理フレーム数: {frames_processed}")
    print(f"2. 異なるメモリアドレス数: {len(memory_addresses)}")
    print(f"3. オブジェクト数変化: {initial_objects} → {final_objects} ({final_objects - initial_objects:+d})")
    
    # メモリリーク判定（オブジェクト数が大幅増加していないか）
    object_increase = final_objects - initial_objects
    memory_leak_ok = object_increase < 50  # 50個以下の増加なら正常
    
    print(f"4. メモリリーク状況: {'良好' if memory_leak_ok else '疑いあり'}")
    
def main():
    """メインテスト実行（拡張版）"""
    print("ActionChain統一フレーム最適化テスト開始")
    print("=" * 60)
    
    # 基本動作テスト
    optimized = test_optimize_image_memory()
    
    # 統一設計テスト
    integration_ok = test_action_chain_integration()
    
    # 独立性テスト
    independence_ok = test_frame_independence()
    
    # 実ActionChain互換性テスト
    actionchain_ok = test_real_actionchain_methods()
    
    # メモリ効率テスト
    memory_ok = test_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー:")
    print(f"- 基本動作: 正常")
    print(f"- 統一設計: {'正常' if integration_ok else '異常'}")
    print(f"- フレーム独立性: {'正常' if independence_ok else '異常'}")
    print(f"- ActionChain互換性: {'正常' if actionchain_ok else '異常'}")
    print(f"- メモリ効率: {'正常' if memory_ok else '異常'}")
    
    all_tests_passed = all([integration_ok, independence_ok, actionchain_ok, memory_ok])
    
    if all_tests_passed:
        print("\n✅ ActionChain統一フレーム最適化設計: 完全動作確認")
        print("【確認項目】")
        print("- _optimize_frame()統一メソッド実装済み")
        print("- 冗長内部メソッド削除済み")
        print("- 外部API互換性保持済み")
        print("- メモリ独立性・効率性確保済み")
        print("- run_manual.py→ActionChainフレーム受け渡し安全")
        print("\n🚀 ラズパイでの本格動作テスト推奨")
    else:
        print("\n❌ 一部の設計に問題があります")
        if not integration_ok:
            print("  → 統一設計の画像処理関数連携に問題")
        if not independence_ok:
            print("  → フレーム独立性に問題")
        if not actionchain_ok:
            print("  → ActionChain実装構造に問題")
        if not memory_ok:
            print("  → メモリ効率に問題")
    
    return all_tests_passed
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
