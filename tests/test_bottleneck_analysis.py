#!/usr/bin/env python3
"""
🔍 control.py ボトルネック分析ツール
============================================================
実際の制御ループで使用される画像処理関数の詳細性能分析
"""

import time
import cv2
import numpy as np
import cProfile
import pstats
import statistics
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from nnspike import constants
from nnspike.utils.control import (
    get_line_edges_at_y, 
    find_bottle_center, 
    find_blue_target_center, 
    get_virtual_line_target_x,
    control_preprocess_image,
    get_color_mask
)

class ControlBottleneckAnalyzer:
    """control.py実関数ボトルネック分析"""
    
    def __init__(self):
        self.test_frames = []
        self.results = {}
        
    def generate_test_frames(self, num_frames=10):
        """テスト用フレーム生成（640x480）"""
        print("📷 テスト用フレーム生成中...")
        
        for i in range(num_frames):
            # 実際のコースに近い画像を生成
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # 黒ライン追加
            frame[240:250, 250:390] = [20, 20, 20]  # 水平黒ライン
            frame[200:350, 320:330] = [20, 20, 20]  # 垂直黒ライン
            
            # 青色ターゲット追加
            frame[150:200, 400:450] = [200, 150, 100]  # 青色領域
            
            # 赤色ボトル追加
            frame[300:350, 500:550] = [100, 100, 200]  # 赤色領域
            
            self.test_frames.append(frame)
        
        print(f"  ✅ {num_frames}フレーム生成完了")

    def measure_function_performance(self, func, func_name, *args, num_runs=50):
        """関数実行時間測定"""
        print(f"\n🧪 {func_name} 性能測定 ({num_runs}回実行)")
        
        execution_times = []
        successful_runs = 0
        
        # ウォームアップ
        for _ in range(5):
            try:
                func(*args)
            except:
                pass
        
        # 実測定
        for i in range(num_runs):
            try:
                start_time = time.perf_counter()
                result = func(*args)
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000  # ms
                execution_times.append(execution_time)
                successful_runs += 1
                
                if (i + 1) % 10 == 0:
                    avg_recent = statistics.mean(execution_times[-10:])
                    print(f"  進行状況: {i+1}/{num_runs} (直近10回平均: {avg_recent:.2f}ms)")
                    
            except Exception as e:
                print(f"  ⚠️ {func_name} 実行エラー: {e}")
        
        if execution_times:
            avg_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            
            print(f"\n📈 {func_name} 結果:")
            print(f"  成功率: {successful_runs/num_runs*100:.1f}% ({successful_runs}/{num_runs})")
            print(f"  平均実行時間: {avg_time:.2f}ms")
            print(f"  中央値: {median_time:.2f}ms")
            print(f"  最速: {min_time:.2f}ms")
            print(f"  最遅: {max_time:.2f}ms")
            print(f"  標準偏差: {std_dev:.2f}ms")
            
            # ボトルネック判定
            if avg_time > 10:
                print(f"  🔥 HIGH IMPACT: 60msサイクルへの大きな影響")
            elif avg_time > 5:
                print(f"  ⚠️ MEDIUM IMPACT: 注意が必要")
            elif avg_time > 1:
                print(f"  📊 LOW IMPACT: 軽微な影響")
            else:
                print(f"  ✅ MINIMAL IMPACT: 影響極小")
            
            self.results[func_name] = {
                'avg_time': avg_time,
                'median_time': median_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_dev': std_dev,
                'success_rate': successful_runs/num_runs*100
            }
        else:
            print(f"  ❌ {func_name}: 測定データなし")

    def measure_preprocess_variations(self):
        """control_preprocess_image の各パラメータ組み合わせ性能測定"""
        print(f"\n🔬 control_preprocess_image バリエーション分析")
        
        test_frame = self.test_frames[0]
        
        # 各種前処理パターン
        preprocess_patterns = [
            ("最小処理", {
                'grayscale': True
            }),
            ("基本処理", {
                'grayscale': True,
                'blur_type': 'gaussian',
                'blur_ksize': 5,
                'binarize_mode': 'binary_inv',
                'binarize_value': 80
            }),
            ("重い処理", {
                'use_hsv': False,
                'grayscale': True,
                'clahe': True,
                'blur_type': 'median',
                'blur_ksize': 7,
                'binarize_mode': 'otsu',
                'noise_removal': ['dilate', 'close7x7']
            }),
            ("最重処理", {
                'use_hsv': True,
                'grayscale': True,
                'clahe': True,
                'clahe_clipLimit': 3.0,
                'blur_type': 'median',
                'blur_ksize': 11,
                'binarize_mode': 'otsu',
                'noise_removal': ['dilate7x3', 'close11x11', 'open3x3']
            })
        ]
        
        for pattern_name, params in preprocess_patterns:
            self.measure_function_performance(
                control_preprocess_image,
                f"preprocess_{pattern_name}",
                test_frame,
                **params
            )

    def profile_heavy_functions(self):
        """重い関数の詳細プロファイリング"""
        print(f"\n🔍 詳細プロファイリング実行")
        
        test_frame = self.test_frames[0]
        
        # プロファイリング対象関数
        profile_targets = [
            ('get_virtual_line_target_x', lambda: get_virtual_line_target_x(test_frame)),
            ('find_blue_target_center', lambda: find_blue_target_center(test_frame)),
            ('find_bottle_center_red', lambda: find_bottle_center(test_frame, "red", roi=constants.ROI_COLOR)),
            ('get_line_edges_at_y', lambda: get_line_edges_at_y(test_frame, constants.ROI_CNN, 470)),
            ('color_mask_blue', lambda: get_color_mask(test_frame, "blue", "line"))
        ]
        
        for func_name, func_lambda in profile_targets:
            print(f"\n📊 {func_name} プロファイリング:")
            
            # cProfileでの詳細分析
            profiler = cProfile.Profile()
            profiler.enable()
            
            # 10回実行
            for _ in range(10):
                try:
                    func_lambda()
                except:
                    pass
            
            profiler.disable()
            
            # 結果分析
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # 上位5つの重い関数を表示
            print("  Top 5 重い処理:")
            stats.print_stats(5)

    def analyze_real_loop_simulation(self):
        """実際の制御ループシミュレーション"""
        print(f"\n🎯 実制御ループシミュレーション")
        
        loop_times = []
        
        for i in range(20):
            frame = self.test_frames[i % len(self.test_frames)]
            
            start_time = time.perf_counter()
            
            # 実際のrun_manual.pyでの処理シミュレーション
            try:
                # Mode.FOLLOW_LEFT_EDGE相当
                _, _, _ = get_line_edges_at_y(frame, constants.ROI_CNN, 470)
                
                # Mode.EYE_BLUE相当
                _, _, _ = find_blue_target_center(frame)
                
                # Mode.CARRY_BOTTLE1相当  
                _, _, _ = find_bottle_center(frame, "red", roi=constants.ROI_COLOR)
                
                # Mode.GATE_PASS相当
                _ = get_virtual_line_target_x(frame)
                
            except Exception as e:
                print(f"  ⚠️ ループ{i+1}エラー: {e}")
            
            end_time = time.perf_counter()
            loop_time = (end_time - start_time) * 1000
            loop_times.append(loop_time)
        
        if loop_times:
            avg_loop = statistics.mean(loop_times)
            print(f"\n📈 実制御ループ結果:")
            print(f"  平均ループ時間: {avg_loop:.1f}ms")
            print(f"  最速ループ: {min(loop_times):.1f}ms")
            print(f"  最遅ループ: {max(loop_times):.1f}ms")
            
            # 現実的な最適化可能性
            if avg_loop > 30:
                print(f"  🎯 最適化余地: 大きい ({avg_loop:.1f}ms → 20-25ms目標)")
            elif avg_loop > 20:
                print(f"  📈 最適化余地: 中程度 ({avg_loop:.1f}ms → 15-20ms目標)")
            else:
                print(f"  ✅ 最適化済み: {avg_loop:.1f}ms")

    def run_bottleneck_analysis(self):
        """ボトルネック分析実行"""
        print("🔍 control.py ボトルネック分析開始")
        print("=" * 60)
        
        # テストフレーム生成
        self.generate_test_frames()
        
        test_frame = self.test_frames[0]
        
        print("\n🎯 実際のrun_manual.py使用関数の性能測定")
        print("=" * 50)
        
        # 1. get_line_edges_at_y (最も頻繁に使用)
        self.measure_function_performance(
            get_line_edges_at_y,
            "get_line_edges_at_y",
            test_frame, constants.ROI_CNN, 470
        )
        
        # 2. find_bottle_center (CARRY_BOTTLE系で使用)
        self.measure_function_performance(
            find_bottle_center,
            "find_bottle_center_red",
            test_frame, "red", constants.ROI_COLOR
        )
        
        # 3. find_blue_target_center (EYE_BLUE系で使用)
        self.measure_function_performance(
            find_blue_target_center,
            "find_blue_target_center",
            test_frame
        )
        
        # 4. get_virtual_line_target_x (GATE_PASS系で使用)
        self.measure_function_performance(
            get_virtual_line_target_x,
            "get_virtual_line_target_x",
            test_frame
        )
        
        # 5. get_color_mask (各種色検出で使用)
        self.measure_function_performance(
            get_color_mask,
            "get_color_mask_blue",
            test_frame, "blue", "line"
        )
        
        # 6. control_preprocess_image バリエーション
        self.measure_preprocess_variations()
        
        # 7. 実制御ループシミュレーション
        self.analyze_real_loop_simulation()
        
        # 8. 詳細プロファイリング
        self.profile_heavy_functions()
        
        # 9. 結果サマリー
        self.generate_bottleneck_summary()

    def generate_bottleneck_summary(self):
        """ボトルネック分析サマリー"""
        print(f"\n🏆 ボトルネック分析サマリー")
        print("=" * 60)
        
        if not self.results:
            print("❌ 測定データなし")
            return
        
        # 実行時間順でソート
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['avg_time'], 
            reverse=True
        )
        
        print("📊 実行時間ランキング:")
        total_impact = 0
        
        for i, (func_name, data) in enumerate(sorted_results[:10], 1):
            avg_time = data['avg_time']
            total_impact += avg_time
            
            impact_level = "🔥" if avg_time > 10 else "⚠️" if avg_time > 5 else "📊" if avg_time > 1 else "✅"
            print(f"  {i:2d}. {impact_level} {func_name}: {avg_time:.2f}ms")
        
        print(f"\n🎯 最適化戦略:")
        
        # Top 3のボトルネック特定
        top3_bottlenecks = sorted_results[:3]
        for func_name, data in top3_bottlenecks:
            avg_time = data['avg_time']
            if avg_time > 5:
                print(f"  🎯 {func_name}: {avg_time:.2f}ms")
                print(f"    → この関数の最適化が60ms→30ms達成に重要")
        
        # 全体的な最適化可能性
        print(f"\n💡 現実的な最適化アプローチ:")
        print(f"  • カーネル事前計算: 既に実装済み")
        print(f"  • ROI最適化: 既に各関数で実装済み")
        print(f"  • アルゴリズム改善: 最重要ボトルネック関数に集中")
        print(f"  • 並列処理: V2で失敗、別アプローチ必要")


def main():
    """メイン実行"""
    analyzer = ControlBottleneckAnalyzer()
    try:
        analyzer.run_bottleneck_analysis()
    except KeyboardInterrupt:
        print("\n⏹️ ユーザー中断")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")


if __name__ == "__main__":
    main()
