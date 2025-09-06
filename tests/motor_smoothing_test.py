#!/usr/bin/env python3
"""
Motor Smoothing Test - 車体浮上状態でのモータースムージングテスト

HIGH_SPEED_BASE=98を前提として、以下をテスト：
1. モータースピード差の削減
2. 速度変化のスムージング
3. PID制御への影響最小化

車体を浮かした状態で実行し、モーター単体の特性を分析します。
"""

import time
import math
import sys
import os
from typing import List, Tuple, Optional

# パスを追加してパッケージをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("NumPyがインストールされていません。基本的な統計計算を使用します。")
    HAS_NUMPY = False
    np = None

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

class MotorSmoothingAnalyzer:
    """モータースムージング分析クラス"""
    
    def __init__(self):
        self.et = ETRobot()
        self.test_results = []
        self.base_speed = HIGH_SPEED_BASE  # 98を使用
        
    def collect_baseline_data(self, duration: float = 5.0) -> List[dict]:
        """ベースライン データ収集（スムージングなし）"""
        print(f"ベースラインデータ収集開始 (HIGH_SPEED_BASE={self.base_speed})")
        results = []
        
        # 固定速度でのテスト
        self.et.set_motor_forward_speed(self.base_speed, self.base_speed)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            status = self.et.get_spike_status()
            
            motor_a_speed = status.motors["A"].speed or 0
            motor_b_speed = status.motors["B"].speed or 0
            
            data = {
                'timestamp': time.time() - start_time,
                'motor_a_speed': motor_a_speed,
                'motor_b_speed': motor_b_speed,
                'motor_a_power': status.motors["A"].power or 0,
                'motor_b_power': status.motors["B"].power or 0,
                'speed_diff': abs(motor_a_speed - motor_b_speed),
                'test_type': 'baseline'
            }
            results.append(data)
            time.sleep(0.02)  # 50Hz sampling
            
        self.et.brake()
        return results
    
    def test_speed_ramping(self, target_speed: int, ramp_time: float = 2.0) -> List[dict]:
        """速度ランプアップテスト"""
        print(f"速度ランプアップテスト: 0 -> {target_speed} ({ramp_time}秒)")
        results = []
        
        start_time = time.time()
        while time.time() - start_time < ramp_time:
            elapsed = time.time() - start_time
            current_speed = int((elapsed / ramp_time) * target_speed)
            
            self.et.set_motor_forward_speed(current_speed, current_speed)
            status = self.et.get_spike_status()
            
            motor_a_speed = status.motors["A"].speed or 0
            motor_b_speed = status.motors["B"].speed or 0
            
            data = {
                'timestamp': elapsed,
                'command_speed': current_speed,
                'motor_a_speed': motor_a_speed,
                'motor_b_speed': motor_b_speed,
                'motor_a_power': status.motors["A"].power or 0,
                'motor_b_power': status.motors["B"].power or 0,
                'speed_diff': abs(motor_a_speed - motor_b_speed),
                'test_type': 'ramp_up'
            }
            results.append(data)
            time.sleep(0.02)
            
        self.et.brake()
        return results
    
    def test_step_response(self, speeds: List[int], hold_time: float = 1.0) -> List[dict]:
        """ステップ応答テスト"""
        print(f"ステップ応答テスト: {speeds}")
        results = []
        
        for speed in speeds:
            print(f"  速度設定: {speed}")
            self.et.set_motor_forward_speed(speed, speed)
            
            start_time = time.time()
            while time.time() - start_time < hold_time:
                elapsed = time.time() - start_time
                status = self.et.get_spike_status()
                
                motor_a_speed = status.motors["A"].speed or 0
                motor_b_speed = status.motors["B"].speed or 0
                
                data = {
                    'timestamp': elapsed,
                    'command_speed': speed,
                    'motor_a_speed': motor_a_speed,
                    'motor_b_speed': motor_b_speed,
                    'motor_a_power': status.motors["A"].power or 0,
                    'motor_b_power': status.motors["B"].power or 0,
                    'speed_diff': abs(motor_a_speed - motor_b_speed),
                    'test_type': f'step_{speed}'
                }
                results.append(data)
                time.sleep(0.02)
                
            time.sleep(0.5)  # 安定待ち
            
        self.et.brake()
        return results
    
    def test_smoothed_control(self, target_changes: List[Tuple[int, int]], smoothing_factor: float = 0.8) -> List[dict]:
        """スムージング制御テスト"""
        print(f"スムージング制御テスト (factor={smoothing_factor})")
        results = []
        
        current_left = current_right = 0
        
        for left_target, right_target in target_changes:
            print(f"  目標速度変更: L={left_target}, R={right_target}")
            
            # スムージング処理
            steps = 20  # スムージングステップ数
            for step in range(steps):
                # 指数移動平均的なスムージング
                current_left += (left_target - current_left) * smoothing_factor / steps
                current_right += (right_target - current_right) * smoothing_factor / steps
                
                self.et.set_motor_forward_speed(int(current_left), int(current_right))
                status = self.et.get_spike_status()
                
                motor_a_speed = status.motors["A"].speed or 0
                motor_b_speed = status.motors["B"].speed or 0
                
                data = {
                    'timestamp': time.time(),
                    'command_left': int(current_left),
                    'command_right': int(current_right),
                    'target_left': left_target,
                    'target_right': right_target,
                    'motor_a_speed': motor_a_speed,
                    'motor_b_speed': motor_b_speed,
                    'motor_a_power': status.motors["A"].power or 0,
                    'motor_b_power': status.motors["B"].power or 0,
                    'speed_diff': abs(motor_a_speed - motor_b_speed),
                    'test_type': 'smoothed',
                    'smoothing_step': step
                }
                results.append(data)
                time.sleep(0.05)  # スムージング間隔
                
            time.sleep(1.0)  # 安定待ち
            
        self.et.brake()
        return results
    
    def calculate_mean(self, data: List[float]) -> float:
        """平均計算（NumPy非依存）"""
        return sum(data) / len(data) if data else 0.0
    
    def calculate_std(self, data: List[float]) -> float:
        """標準偏差計算（NumPy非依存）"""
        if not data or len(data) < 2:
            return 0.0
        mean = self.calculate_mean(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return variance ** 0.5
    
    def analyze_results(self, results: List[dict]) -> dict:
        """結果分析"""
        if not results:
            return {}
            
        # 基本統計
        speed_diffs = [r['speed_diff'] for r in results if 'speed_diff' in r]
        motor_a_speeds = [r['motor_a_speed'] for r in results if r['motor_a_speed'] is not None]
        motor_b_speeds = [r['motor_b_speed'] for r in results if r['motor_b_speed'] is not None]
        
        analysis = {
            'total_samples': len(results),
            'avg_speed_diff': self.calculate_mean(speed_diffs),
            'max_speed_diff': max(speed_diffs) if speed_diffs else 0,
            'std_speed_diff': self.calculate_std(speed_diffs),
            'avg_motor_a_speed': self.calculate_mean(motor_a_speeds),
            'avg_motor_b_speed': self.calculate_mean(motor_b_speeds),
            'motor_a_variation': self.calculate_std(motor_a_speeds),
            'motor_b_variation': self.calculate_std(motor_b_speeds)
        }
        
        return analysis
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("=== モータースムージング包括テスト開始 ===")
        print(f"HIGH_SPEED_BASE: {self.base_speed}")
        print("注意: 車体を浮上させた状態で実行してください")
        
        input("準備ができたらEnterキーを押してください...")
        
        all_results = {}
        
        # 1. ベースラインテスト
        print("\n1. ベースラインテスト")
        baseline_results = self.collect_baseline_data()
        all_results['baseline'] = {
            'data': baseline_results,
            'analysis': self.analyze_results(baseline_results)
        }
        
        # 2. ステップ応答テスト
        print("\n2. ステップ応答テスト")
        step_speeds = [30, 60, self.base_speed, 120, 80, 40]
        step_results = self.test_step_response(step_speeds)
        all_results['step_response'] = {
            'data': step_results,
            'analysis': self.analyze_results(step_results)
        }
        
        # 3. ランプアップテスト
        print("\n3. ランプアップテスト")
        ramp_results = self.test_speed_ramping(self.base_speed)
        all_results['ramp_up'] = {
            'data': ramp_results,
            'analysis': self.analyze_results(ramp_results)
        }
        
        # 4. スムージングテスト（軽め）
        print("\n4. スムージングテスト（軽め）")
        smoothed_light_results = self.test_smoothed_control([
            (50, 50), (80, 70), (60, 90), (self.base_speed, self.base_speed)
        ], smoothing_factor=0.3)
        all_results['smoothed_light'] = {
            'data': smoothed_light_results,
            'analysis': self.analyze_results(smoothed_light_results)
        }
        
        # 5. スムージングテスト（強め）
        print("\n5. スムージングテスト（強め）")
        smoothed_heavy_results = self.test_smoothed_control([
            (50, 50), (80, 70), (60, 90), (self.base_speed, self.base_speed)
        ], smoothing_factor=0.8)
        all_results['smoothed_heavy'] = {
            'data': smoothed_heavy_results,
            'analysis': self.analyze_results(smoothed_heavy_results)
        }
        
        # 結果表示
        self.print_comparison(all_results)
        
        return all_results
    
    def print_comparison(self, results: dict):
        """結果比較表示"""
        print("\n=== テスト結果比較 ===")
        print(f"{'テスト名':<20} {'平均速度差':<10} {'最大速度差':<10} {'速度差標準偏差':<12} {'総サンプル数':<10}")
        print("-" * 70)
        
        for test_name, test_data in results.items():
            analysis = test_data['analysis']
            print(f"{test_name:<20} {analysis['avg_speed_diff']:<10.1f} {analysis['max_speed_diff']:<10.1f} {analysis['std_speed_diff']:<12.1f} {analysis['total_samples']:<10}")
        
        # 推奨設定
        print("\n=== 推奨設定 ===")
        baseline_std = results['baseline']['analysis']['std_speed_diff']
        
        best_smoothing = None
        best_improvement = 0
        
        for test_name, test_data in results.items():
            if 'smoothed' in test_name:
                improvement = baseline_std - test_data['analysis']['std_speed_diff']
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_smoothing = test_name
        
        if best_smoothing:
            print(f"最適なスムージング: {best_smoothing}")
            print(f"改善効果: {best_improvement:.1f} (速度差標準偏差の削減)")
        else:
            print("スムージングによる明確な改善は見られませんでした")

def main():
    analyzer = MotorSmoothingAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_test()
        
        # 結果をファイルに保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        import json
        
        # JSONシリアライズのための型変換
        def convert_types(obj):
            if HAS_NUMPY and np is not None:
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
            return obj
        
        # analysisのみをJSONに保存（データが大きすぎるため）
        summary = {test_name: test_data['analysis'] for test_name, test_data in results.items()}
        
        # ファイル保存パスをOSに応じて決定
        storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'storage')
        os.makedirs(storage_dir, exist_ok=True)
        output_file = os.path.join(storage_dir, f'motor_smoothing_test_{timestamp}.json')
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=convert_types)
        
        print(f"\n結果を保存しました: {output_file}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        analyzer.et.stop()

if __name__ == "__main__":
    main()
