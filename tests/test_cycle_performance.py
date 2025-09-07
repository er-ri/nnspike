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


class CyclePerformanceTester:
    """制御サイクル性能測定クラス"""
    
    def __init__(self):
        self.etrobot = None
        self.cap = None
        self.results = {
            'baseline': [],      # 現状60ms
            'optimized_v1': [],  # 画像処理最適化
            'optimized_v2': [],  # 並列処理導入
            'optimized_v3': []   # 全最適化
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
        """最適化v1: 画像処理高速化"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # 2. 画像処理最適化
        # ROI設定 (下半分のみ処理)
        h, w = frame.shape[:2]
        roi_frame = frame[h//2:, :]
        
        # 解像度1/2に縮小
        small_frame = cv2.resize(roi_frame, (w//2, h//4))
        
        # HSV変換 (ROIのみ)
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        
        # 簡易色フィルタリング
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        
        # 簡易輪郭検出
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Spike通信
        if self.etrobot:
            status = self.etrobot.get_spike_status()
        
        # 4. 制御演算
        control_value = self._calculate_control(contours)
        
        # 5. モーター出力
        if self.etrobot and control_value is not None:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
    def optimized_v2_cycle(self):
        """最適化v2: 並列処理導入"""
        start_time = time.perf_counter()
        
        # 1. カメラフレーム取得 (別スレッドから)
        if hasattr(self, 'frame_queue') and not self.frame_queue.empty():
            frame = self.frame_queue.get_nowait()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        # 2. 高速画像処理
        h, w = frame.shape[:2]
        roi_frame = frame[h//2:, :]
        small_frame = cv2.resize(roi_frame, (w//4, h//8))  # さらに縮小
        
        # グレースケール処理のみ
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 簡易閾値処理
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # 重心計算
        moments = cv2.moments(binary)
        control_value = 0
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            control_value = (cx - small_frame.shape[1]//2) * 0.1
        
        # 3. Spike通信 (キャッシュ利用)
        if hasattr(self, 'cached_status'):
            status = self.cached_status
        elif self.etrobot:
            status = self.etrobot.get_spike_status()
            self.cached_status = status
        
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
        """最適化v3: 全最適化適用"""
        start_time = time.perf_counter()
        
        # フレームスキップ実装
        if not hasattr(self, 'skip_counter'):
            self.skip_counter = 0
        
        self.skip_counter += 1
        
        # 2フレームに1回だけ画像処理
        if self.skip_counter % 2 == 0:
            if hasattr(self, 'frame_queue') and not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # 最小限画像処理
                h, w = frame.shape[:2]
                tiny_frame = cv2.resize(frame[h//2:, :], (80, 40))
                gray = cv2.cvtColor(tiny_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                
                # 重心のみ計算
                moments = cv2.moments(binary)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    self.last_control = (cx - 40) * 0.05
                else:
                    self.last_control = 0
        
        # 前回の制御値を使用
        control_value = getattr(self, 'last_control', 0)
        
        # Spike通信スキップ (10サイクルに1回)
        if self.skip_counter % 10 == 0 and self.etrobot:
            self.cached_status = self.etrobot.get_spike_status()
        
        # モーター出力
        if self.etrobot:
            base_speed = 30
            steering_offset = int(abs(control_value * 50))
            left_speed = max(0, min(100, base_speed - steering_offset if control_value > 0 else base_speed + steering_offset))
            right_speed = max(0, min(100, base_speed + steering_offset if control_value > 0 else base_speed - steering_offset))
            self.etrobot.set_motor_forward_speed(left_speed, right_speed)
        
        cycle_time = (time.perf_counter() - start_time) * 1000
        return cycle_time
    
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
            
            # テスト2: 画像処理最適化
            self.test_cycle_performance("Optimized_V1", self.optimized_v1_cycle, 50)
            
            # テスト3: 並列処理導入
            self.run_background_capture()  # バックグラウンド取得開始
            time.sleep(1)  # 安定化待機
            self.test_cycle_performance("Optimized_V2", self.optimized_v2_cycle, 50)
            
            # テスト4: 全最適化
            self.test_cycle_performance("Optimized_V3", self.optimized_v3_cycle, 50)
            
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
            ('optimized_v1', '画像処理最適化'),
            ('optimized_v2', '並列処理導入'),
            ('optimized_v3', '全最適化適用')
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
