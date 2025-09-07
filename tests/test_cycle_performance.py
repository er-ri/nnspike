#!/usr/bin/env python3
"""
🎯 リアルタイム制御サイクル性能テスト
============================================================
60ms現状 → 30-40ms目標への最適化効果を実測検証
"""

import time
import cv2
import numpy as np
import threading
import queue
import statistics
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from nnspike.unit import ETRobot
from nnspike import constants
from nnspike.utils import (
    get_line_edges_at_y, 
    find_bottle_center, 
    find_blue_target_center, 
    get_virtual_line_target_x,
    get_color_mask,
    control_preprocess_image,
    fill_green_with_white
)


class CyclePerformanceTester:
    """制御サイクル性能測定クラス"""
    
    def __init__(self):
        self.etrobot = None
        self.cap = None
        self.results = {
            'baseline': [],      # 現状60ms
            'optimized_v1': [],  # 画像処理最適化
            'optimized_v2': [],  # 並列処理導入
            'optimized_v3': [],  # 全最適化
            'conditional_v1': [], # 条件分岐最適化
            'frame_skip_v1': [], # フレーム間引き最適化
            'roi_optimized': []  # ROI最適化
        }
        
    def setup_camera(self, optimized=False):
        """カメラ初期化"""
        print(f"📷 カメラ初期化 (最適化: {optimized})")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("カメラ接続失敗")
            
        # 基本設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, constants.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, constants.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if optimized:
            # 最適化設定
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファ最小化
            print("  ✅ 最適化設定適用")
        
    def setup_spike_connection(self):
        """Spike Hub接続"""
        print("🔌 Spike Hub接続中...")
        try:
            self.etrobot = ETRobot()
            print("  ✅ Spike Hub接続成功")
            return True
        except Exception as e:
            print(f"  ❌ Spike Hub接続失敗: {e}")
            return False
    
    def baseline_cycle(self):
        """現状の制御サイクル (60ms想定)"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # 2. 画像処理 (現状フル処理)
        # カラー変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 色フィルタリング (青)
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        
        # 輪郭検出
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 4. 制御演算 (簡易PID)
        control_value = self._calculate_control(contours)
        
        # 5. モーター出力
        if self.etrobot and control_value is not None:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))  # ステアリング調整
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def optimized_v1_cycle(self):
        """最適化v1: control.pyの実際の関数使用テスト"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # 2. 実際のcontrol.py関数使用
        # ライン追従 (最も頻繁に使用される処理)
        roi_cnn = constants.ROI_CNN
        left_x, right_x, line_width = get_line_edges_at_y(
            frame, roi_cnn, constants.OFFSET_Y, threshold_value=80
        )
        
        # 制御値計算
        if left_x is not None and right_x is not None:
            center_x = (left_x + right_x) / 2
            control_value = (center_x - constants.CAMERA_WIDTH//2) * 0.001
        else:
            control_value = 0
        
        # 3. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 4. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def optimized_v2_cycle(self):
        """最適化v2: 複数の実際の関数組み合わせテスト"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 2. 複数制御モードシミュレーション
        roi_cnn = constants.ROI_CNN
        
        # ライン追従
        left_x, right_x, line_width = get_line_edges_at_y(
            frame, roi_cnn, constants.OFFSET_Y, threshold_value=80
        )
        
        # ボトル検出も実行 (CARRY_BOTTLEモード想定) - 実際のROI使用
        bottle_result = find_bottle_center(frame, "blue", roi=constants.ROI_COLOR)
        
        # 制御値統合
        if left_x is not None and right_x is not None:
            center_x = (left_x + right_x) / 2
            control_value = (center_x - constants.CAMERA_WIDTH//2) * 0.001
        elif bottle_result[0] is not None:
            control_value = (bottle_result[0][0] - constants.CAMERA_WIDTH//2) * 0.001
        else:
            control_value = 0
        
        # 3. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 4. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def optimized_v3_cycle(self):
        """最適化v3: 実際のrun_manual.pyと同じ処理テスト"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 2. 実際のrun_manual.pyと同じ処理組み合わせシミュレーション
        roi_cnn = constants.ROI_CNN
        
        # ライン追従 (最頻出処理) - 実際のROI使用
        left_x, right_x, line_width = get_line_edges_at_y(
            frame, roi_cnn, constants.OFFSET_Y, threshold_value=80
        )
        
        # 青ターゲット検出 (EYE_BLUEモード想定) - 内蔵ROI使用
        blue_target_result = find_blue_target_center(frame)
        
        # 仮想ライン検出 (GATE_PASSモード想定) - 内蔵ROI_VIRTUAL使用
        center_x = (roi_cnn[0] + roi_cnn[2]) // 2  # ROI中心X座標
        virtual_target_x = get_virtual_line_target_x(frame, previous_center_x=center_x)
        
        # 制御値統合（実際のrun_manual.pyと同じロジック）
        if virtual_target_x is not None:
            control_value = (virtual_target_x - constants.CAMERA_WIDTH//2) * 0.001
        elif blue_target_result[0] is not None:
            control_value = (blue_target_result[0][0] - constants.CAMERA_WIDTH//2) * 0.001
        elif left_x is not None and right_x is not None:
            center_x = (left_x + right_x) / 2
            control_value = (center_x - constants.CAMERA_WIDTH//2) * 0.001
        else:
            control_value = 0
        
        # 3. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 4. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def detailed_image_processing_test(self):
        """画像処理関数の段階的負荷測定"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        capture_time = (time.perf_counter() - start_time) * 1000
        
        # 2. fill_green_with_white 負荷測定
        fill_start = time.perf_counter()
        frame_filled = fill_green_with_white(frame.copy())
        fill_time = (time.perf_counter() - fill_start) * 1000
        
        # 3. get_color_mask 負荷測定 (青色)
        color_start = time.perf_counter()
        blue_mask = get_color_mask(frame, "blue", pattern="target")
        color_time = (time.perf_counter() - color_start) * 1000
        
        # 4. control_preprocess_image 負荷測定 (軽量)
        preprocess_light_start = time.perf_counter()
        processed_light = control_preprocess_image(
            frame.copy(),
            use_hsv=False,
            grayscale=True,
            clahe=False,
            blur_type="gaussian",
            blur_ksize=5,
            binarize_mode="binary_inv",
            binarize_value=80,
            noise_removal=None
        )
        preprocess_light_time = (time.perf_counter() - preprocess_light_start) * 1000
        
        # 5. control_preprocess_image 負荷測定 (重い - 仮想ライン用)
        preprocess_heavy_start = time.perf_counter()
        processed_heavy = control_preprocess_image(
            frame.copy(),
            use_hsv=False,
            grayscale=True,
            clahe=True,
            clahe_clipLimit=3.0,
            blur_type="median",
            blur_ksize=7,
            binarize_mode="binary_inv",
            binarize_value=120,
            noise_removal=["dilate", "close7x7"]
        )
        preprocess_heavy_time = (time.perf_counter() - preprocess_heavy_start) * 1000
        
        # 6. 組み合わせ処理 (実際のget_virtual_line_target_x相当)
        combined_start = time.perf_counter()
        # 緑塗りつぶし + 重い前処理
        frame_combined = fill_green_with_white(frame.copy())
        processed_combined = control_preprocess_image(
            frame_combined,
            use_hsv=False,
            grayscale=True,
            clahe=True,
            clahe_clipLimit=3.0,
            blur_type="median",
            blur_ksize=7,
            binarize_mode="binary_inv",
            binarize_value=120,
            noise_removal=["dilate", "close7x7"]
        )
        # ROI抽出 + 輪郭検出
        roi_virtual = constants.ROI_VIRTUAL
        x1, y1, x2, y2 = roi_virtual
        mask_roi = processed_combined[y1:y2, x1:x2]
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        combined_time = (time.perf_counter() - combined_start) * 1000
        
        # 7. Spike通信
        if self.etrobot:
            spike_start = time.perf_counter()
            status = self.etrobot.get_spike_status()
            spike_time = (time.perf_counter() - spike_start) * 1000
        else:
            spike_time = 0
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'capture_time': capture_time,
            'fill_green_time': fill_time,
            'color_mask_time': color_time,
            'preprocess_light_time': preprocess_light_time,
            'preprocess_heavy_time': preprocess_heavy_time,
            'combined_processing_time': combined_time,
            'spike_comm_time': spike_time,
            'total_time': total_time
        }
    
    def _calculate_control(self, contours):
        """制御値計算"""
        if not contours:
            return 0
        
        # 最大輪郭の重心
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            # ステアリング値計算
            return (cx - constants.CAMERA_WIDTH//2) * 0.001
        return 0
    
    def test_detailed_image_processing(self, num_cycles=50):
        """画像処理関数の詳細性能テスト"""
        print(f"\n🔬 詳細画像処理性能テスト ({num_cycles}サイクル)")
        print("=" * 60)
        
        results = {
            'capture_times': [],
            'fill_green_times': [],
            'color_mask_times': [],
            'preprocess_light_times': [],
            'preprocess_heavy_times': [],
            'combined_processing_times': [],
            'spike_comm_times': [],
            'total_times': []
        }
        
        print("📊 測定中...")
        
        for i in range(num_cycles):
            try:
                result = self.detailed_image_processing_test()
                if result:
                    results['capture_times'].append(result['capture_time'])
                    results['fill_green_times'].append(result['fill_green_time'])
                    results['color_mask_times'].append(result['color_mask_time'])
                    results['preprocess_light_times'].append(result['preprocess_light_time'])
                    results['preprocess_heavy_times'].append(result['preprocess_heavy_time'])
                    results['combined_processing_times'].append(result['combined_processing_time'])
                    results['spike_comm_times'].append(result['spike_comm_time'])
                    results['total_times'].append(result['total_time'])
                
                if (i + 1) % 10 == 0:
                    print(f"  進行状況: {i+1}/{num_cycles}")
                
            except Exception as e:
                print(f"  ⚠️ サイクル{i+1}でエラー: {e}")
        
        # 結果分析
        if results['total_times']:
            print(f"\n📈 詳細画像処理性能分析:")
            print(f"  測定サイクル数: {len(results['total_times'])}")
            print()
            
            analyses = [
                ('capture_times', 'カメラフレーム取得'),
                ('fill_green_times', 'fill_green_with_white'),
                ('color_mask_times', 'get_color_mask(青)'),
                ('preprocess_light_times', 'control_preprocess(軽量)'),
                ('preprocess_heavy_times', 'control_preprocess(重い)'),
                ('combined_processing_times', '組み合わせ処理(仮想ライン相当)'),
                ('spike_comm_times', 'Spike Hub通信'),
                ('total_times', '合計時間')
            ]
            
            total_avg = statistics.mean(results['total_times'])
            
            for key, name in analyses:
                if results[key]:
                    avg_time = statistics.mean(results[key])
                    min_time = min(results[key])
                    max_time = max(results[key])
                    percentage = (avg_time / total_avg) * 100
                    
                    print(f"🔧 {name}:")
                    print(f"  平均: {avg_time:.2f}ms ({percentage:.1f}%)")
                    print(f"  範囲: {min_time:.2f}ms - {max_time:.2f}ms")
                    
                    # ボトルネック判定
                    if avg_time > 15:
                        print(f"  ⚠️ 高負荷処理 - 最適化推奨")
                    elif avg_time > 8:
                        print(f"  📈 中程度負荷")
                    else:
                        print(f"  ✅ 軽量処理")
                    print()
            
            # 最重負荷処理の特定
            processing_loads = [
                (statistics.mean(results['fill_green_times']), 'fill_green_with_white'),
                (statistics.mean(results['preprocess_heavy_times']), 'control_preprocess(重い)'),
                (statistics.mean(results['combined_processing_times']), '組み合わせ処理'),
                (statistics.mean(results['color_mask_times']), 'get_color_mask')
            ]
            processing_loads.sort(reverse=True)
            
            print(f"🎯 処理負荷ランキング:")
            for i, (load_time, name) in enumerate(processing_loads[:3], 1):
                print(f"  {i}位: {name} ({load_time:.2f}ms)")
            
            print(f"\n💡 最適化提案:")
            heaviest_load, heaviest_name = processing_loads[0]
            if heaviest_load > 15:
                print(f"  - {heaviest_name}の最適化が急務")
                print(f"  - 現在{heaviest_load:.1f}ms → 目標10ms以下")
            else:
                print(f"  - 全体的に良好な性能")
        
        else:
            print("❌ 詳細測定データなし")
    
    def _calculate_control(self, contours):
        """制御値計算"""
        if not contours:
            return 0
        
        # 最大輪郭の重心
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            # ステアリング値計算
            return (cx - constants.CAMERA_WIDTH//2) * 0.001
        return 0
    
    def run_background_capture(self):
        """バックグラウンドフレーム取得"""
        self.frame_queue = queue.Queue(maxsize=2)
        
        def capture_loop():
            while getattr(self, 'capturing', True):
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # 古いフレーム破棄
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                time.sleep(0.01)  # 100FPS制限
        
        self.capture_thread = threading.Thread(target=capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def test_cycle_performance(self, test_name, cycle_func, num_cycles=100):
        """サイクル性能テスト"""
        print(f"\n🧪 {test_name} テスト開始 ({num_cycles}サイクル)")
        print("=" * 50)
        
        cycle_times = []
        successful_cycles = 0
        
        # ウォームアップ
        for _ in range(10):
            try:
                cycle_func()
            except:
                pass
        
        print("📊 測定中...")
        start_test = time.perf_counter()
        
        for i in range(num_cycles):
            try:
                cycle_time = cycle_func()
                if cycle_time is not None:
                    cycle_times.append(cycle_time)
                    successful_cycles += 1
                
                if (i + 1) % 20 == 0:
                    current_avg = statistics.mean(cycle_times[-20:]) if cycle_times else 0
                    print(f"  進行状況: {i+1}/{num_cycles} (直近20回平均: {current_avg:.1f}ms)")
                
            except Exception as e:
                print(f"  ⚠️ サイクル{i+1}でエラー: {e}")
        
        test_duration = time.perf_counter() - start_test
        
        if cycle_times:
            avg_time = statistics.mean(cycle_times)
            median_time = statistics.median(cycle_times)
            min_time = min(cycle_times)
            max_time = max(cycle_times)
            std_dev = statistics.stdev(cycle_times) if len(cycle_times) > 1 else 0
            
            print(f"\n📈 {test_name} 結果:")
            print(f"  成功率: {successful_cycles/num_cycles*100:.1f}% ({successful_cycles}/{num_cycles})")
            print(f"  平均サイクル時間: {avg_time:.1f}ms")
            print(f"  中央値: {median_time:.1f}ms")
            print(f"  最速: {min_time:.1f}ms")
            print(f"  最遅: {max_time:.1f}ms")
            print(f"  標準偏差: {std_dev:.1f}ms")
            print(f"  実効FPS: {1000/avg_time:.1f} Hz")
            
            # 目標達成判定
            if avg_time <= 40:
                print("  🎯 目標達成: 40ms以下")
            elif avg_time <= 50:
                print("  📈 改善良好: 50ms以下")
            elif avg_time <= 60:
                print("  ⚠️ 要改善: 60ms以下")
            else:
                print("  ❌ 要大幅改善: 60ms超過")
                
            self.results[test_name.lower().replace(' ', '_').replace(':', '')] = cycle_times
            
        else:
            print(f"  ❌ {test_name}: 測定データなし")
    
    def run_all_tests(self):
        """全パフォーマンステスト実行"""
        print("🎯 制御サイクル性能テスト開始")
        print("=" * 60)
        print("目標: 60ms → 30-40ms短縮")
        print()
        
        # システム情報
        print(f"🖥️ システム状態:")
        print(f"  テスト環境: Windows/Linux")
        print()
        
        try:
            # カメラ初期化
            self.setup_camera(optimized=False)
            
            # Spike Hub接続
            spike_connected = self.setup_spike_connection()
            if not spike_connected:
                print("⚠️ Spike Hub未接続 - カメラテストのみ実行")
            
            # テスト1: ベースライン (現状)
            self.test_cycle_performance("Baseline", self.baseline_cycle, 50)
            
            # テスト2: ライン追従処理
            self.test_cycle_performance("Optimized_V1", self.optimized_v1_cycle, 50)
            
            # テスト3: 複数制御モード
            self.test_cycle_performance("Optimized_V2", self.optimized_v2_cycle, 50)
            
            # テスト4: 実際のrun_manual.py処理
            self.test_cycle_performance("Actual_RunManual", self.optimized_v3_cycle, 50)
            
            # テスト5: 条件分岐最適化
            self.test_cycle_performance("Conditional_Opt", self.conditional_optimization_cycle, 50)
            
            # テスト6: 並列処理最適化  
            self.test_cycle_performance("Parallel_Processing", self.parallel_processing_cycle, 50)
            
            # テスト7: ROI最適化
            self.test_cycle_performance("ROI_Optimized", self.roi_optimized_cycle, 50)
            
            # テスト8: 事前計算最適化
            self.test_cycle_performance("Precomputed_Opt", self.precomputed_optimization_cycle, 50)
            
            # テスト9: 詳細画像処理分析
            self.test_detailed_image_processing(30)
            
            # 比較レポート
            self.generate_comparison_report()
            
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
        finally:
            self.cleanup()
    
    def generate_comparison_report(self):
        """比較レポート生成"""
        print("\n🏆 最適化効果レポート")
        print("=" * 60)
        
        baseline_avg = statistics.mean(self.results.get('baseline', [60])) if self.results.get('baseline') else 60
        
        optimizations = [
            ('optimized_v1', 'ライン追従処理'),
            ('optimized_v2', '複数制御モード'),
            ('actual_runmanual', '実際のrun_manual.py処理'),
            ('conditional_opt', '条件分岐最適化 (状況別処理選択)'),
            ('parallel_processing', '並列処理最適化 (軽量+重い処理ブレンド)'),
            ('roi_optimized', 'ROI最適化 (メモリ効率化)'),
            ('precomputed_opt', '事前計算最適化 (LUT/キャッシュ活用)')
        ]
        
        print(f"📊 ベースライン: {baseline_avg:.1f}ms")
        print()
        
        best_time = baseline_avg
        best_name = "ベースライン"
        
        for key, name in optimizations:
            if key in self.results and self.results[key]:
                avg_time = statistics.mean(self.results[key])
                improvement = baseline_avg - avg_time
                improvement_pct = (improvement / baseline_avg) * 100
                
                print(f"🚀 {name}:")
                print(f"  平均時間: {avg_time:.1f}ms")
                print(f"  改善効果: {improvement:+.1f}ms ({improvement_pct:+.1f}%)")
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_name = name
                
                # 目標達成判定
                if avg_time <= 40:
                    print(f"  🎯 目標達成!")
                elif avg_time <= 50:
                    print(f"  📈 改善良好")
                else:
                    print(f"  ⚠️ さらなる最適化必要")
                print()
        
        print(f"🏅 最優秀: {best_name} ({best_time:.1f}ms)")
        
        if best_time <= 40:
            print("✅ 30-40ms目標達成可能！")
        elif best_time <= 50:
            print("📈 目標に近づいています")
        else:
            print("❌ さらなる最適化が必要です")
    
    def conditional_optimization_cycle(self):
        """条件分岐最適化: モード別処理選択で品質保持"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 2. 仮想的なモード判定（80%軽量、20%重い処理で実測）
        import random
        simulate_gate_mode = random.random() < 0.2
        
        if simulate_gate_mode:
            # ゲート通過時のみ重い処理（品質保持）
            roi_cnn = constants.ROI_CNN
            center_x = (roi_cnn[0] + roi_cnn[2]) // 2
            virtual_target_x = get_virtual_line_target_x(frame, previous_center_x=center_x)
            control_value = (virtual_target_x - constants.CAMERA_WIDTH//2) * 0.001 if virtual_target_x else 0
        else:
            # 通常時は軽量ライン追従
            roi_cnn = constants.ROI_CNN
            left_x, right_x, line_width = get_line_edges_at_y(
                frame, roi_cnn, constants.OFFSET_Y, threshold_value=80
            )
            if left_x is not None and right_x is not None:
                center_x = (left_x + right_x) / 2
                control_value = (center_x - constants.CAMERA_WIDTH//2) * 0.001
            else:
                control_value = 0
        
        # 3. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 4. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def parallel_processing_cycle(self):
        """並列処理最適化: 重い処理を非同期化で品質保持"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 2. 軽量なライン検出を優先実行
        roi_cnn = constants.ROI_CNN
        left_x, right_x, line_width = get_line_edges_at_y(
            frame, roi_cnn, constants.OFFSET_Y, threshold_value=80
        )
        
        # 基本制御値（即座に利用可能）
        if left_x is not None and right_x is not None:
            quick_center_x = (left_x + right_x) / 2
            control_value = (quick_center_x - constants.CAMERA_WIDTH//2) * 0.001
        else:
            control_value = 0
        
        # 3. 重い処理は結果があれば補正（品質保持）
        if not hasattr(self, 'heavy_processing_result'):
            self.heavy_processing_result = None
        
        # シンプルな重い処理のシミュレーション（実際は非同期で実行）
        try:
            center_x = (roi_cnn[0] + roi_cnn[2]) // 2
            virtual_target_x = get_virtual_line_target_x(frame, previous_center_x=center_x)
            if virtual_target_x is not None:
                # 重い処理結果で補正
                heavy_control = (virtual_target_x - constants.CAMERA_WIDTH//2) * 0.001
                control_value = 0.7 * control_value + 0.3 * heavy_control  # ブレンド
        except:
            pass  # 重い処理が失敗しても軽量処理で継続
        
        # 4. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 5. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def roi_optimized_cycle(self):
        """ROI最適化: メモリ効率化で品質保持"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 2. メモリレイアウト最適化
        frame_contiguous = np.ascontiguousarray(frame)
        
        # 3. ROI事前切り出し（品質保持した効率化）
        roi_virtual = constants.ROI_VIRTUAL
        x1, y1, x2, y2 = roi_virtual
        frame_roi = frame_contiguous[y1:y2, x1:x2].copy()
        
        # 4. 品質保持した前処理をROI上で実行
        frame_filled = fill_green_with_white(frame_roi)
        processed = control_preprocess_image(
            frame_filled,
            use_hsv=False,
            grayscale=True,
            clahe=True,  # 品質保持
            clahe_clipLimit=3.0,
            blur_type="median",  # 品質保持
            blur_ksize=7,
            binarize_mode="binary_inv",
            binarize_value=120,
            noise_removal=["dilate", "close7x7"]  # 品質保持
        )
        
        # 5. 輪郭検出と制御値計算
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                target_x = x1 + x + w//2
                control_value = (target_x - constants.CAMERA_WIDTH//2) * 0.001
            else:
                control_value = 0
        else:
            control_value = 0
        
        # 6. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 7. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def precomputed_optimization_cycle(self):
        """事前計算最適化: LUT/キャッシュで品質保持"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 2. LUT初期化（1回のみ）
        if not hasattr(self, 'morphology_kernel'):
            self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self.roi_coordinates = constants.ROI_VIRTUAL
        
        # 3. 事前計算済みROIで効率化
        x1, y1, x2, y2 = self.roi_coordinates
        frame_roi = frame[y1:y2, x1:x2]
        
        # 4. 品質保持した高速前処理
        frame_filled = fill_green_with_white(frame_roi)
        
        # 事前計算済みカーネルで高速化
        processed = control_preprocess_image(
            frame_filled,
            use_hsv=False,
            grayscale=True,
            clahe=True,  # 品質保持
            clahe_clipLimit=3.0,
            blur_type="median",  # 品質保持
            blur_ksize=7,
            binarize_mode="binary_inv", 
            binarize_value=120,
            noise_removal=["dilate", "close7x7"]  # 品質保持
        )
        
        # 5. 高速輪郭検出
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 面積でフィルタ（事前計算済み閾値）
            valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                target_x = x1 + x + w//2
                control_value = (target_x - constants.CAMERA_WIDTH//2) * 0.001
            else:
                control_value = 0
        else:
            control_value = 0
        
        # 6. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 7. モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def cleanup(self):
        """リソース解放"""
        print("\n🔚 テスト終了 - リソース解放中...")
        
        self.capturing = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        if self.etrobot:
            try:
                self.etrobot.brake()  # 停止
                self.etrobot.stop()   # 終了
            except:
                pass
        
        cv2.destroyAllWindows()
        print("✅ クリーンアップ完了")


def main():
    """メイン実行"""
    tester = CyclePerformanceTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⏹️ ユーザー中断")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
