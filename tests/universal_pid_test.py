#!/usr/bin/env python3
"""
汎用PID設定テスト - HIGH_SPEED_BASE 70/98 両対応

HIGH_SPEED_BASE=70と98の両方で最適に動作する
汎用的なPID設定を見つけるテスト
"""

import time
import math
import sys
import os
from typing import List, Dict, Tuple

# パスを追加してパッケージをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class UniversalPIDTester:
    """汎用PID設定テストクラス"""
    
    def __init__(self):
        self.et = ETRobot()
        self.test_results = []
        
    def calculate_stats(self, data: List[float]) -> Dict:
        """統計計算"""
        if not data:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std = math.sqrt(variance)
        
        return {
            'mean': mean,
            'std': std,
            'min': min(data),
            'max': max(data),
            'count': n
        }
    
    def test_universal_pid(self, kp: float, ki: float, kd: float, 
                          output_limits: Tuple[float, float], 
                          base_speed: int, duration: float = 3.0) -> Dict:
        """汎用PID制御テスト"""
        print(f"🔬 汎用PID Kp={kp}, Ki={ki}, Kd={kd}, limits={output_limits}")
        print(f"   BASE_SPEED={base_speed}")
        
        # PID制御器を設定
        pid = PIDController(
            Kp=kp, Ki=ki, Kd=kd,
            setpoint=0,
            output_limits=output_limits
        )
        
        speed_diffs = []
        corrections = []
        actual_speeds_a = []
        actual_speeds_b = []
        
        # 相対位置をリセット
        self.et.set_motor_relative_position(0, 0)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            status = self.et.get_spike_status()
            
            pos_a = status.motors["A"].relative_position
            pos_b = status.motors["B"].relative_position
            speed_a = abs(status.motors["A"].speed or 0)
            speed_b = abs(status.motors["B"].speed or 0)
            
            if pos_a is not None and pos_b is not None:
                # 位置偏差を計算
                position_error = pos_a - pos_b
                
                # PID制御で補正
                correction = pid.update(position_error)
                corrections.append(abs(correction))
                
                # 補正を適用した速度設定
                left_speed = int(max(0, min(255, base_speed - correction)))
                right_speed = int(max(0, min(255, base_speed + correction)))
                
                self.et.set_motor_forward_speed(left_speed, right_speed)
                
                # データ記録
                if speed_a > 0 and speed_b > 0:
                    speed_diff = abs(speed_a - speed_b)
                    speed_diffs.append(speed_diff)
                    actual_speeds_a.append(speed_a)
                    actual_speeds_b.append(speed_b)
            
            time.sleep(0.05)  # 20Hz
        
        self.et.brake()
        
        # 統計計算
        speed_stats = self.calculate_stats(speed_diffs)
        correction_stats = self.calculate_stats(corrections)
        actual_speed_a_stats = self.calculate_stats(actual_speeds_a)
        actual_speed_b_stats = self.calculate_stats(actual_speeds_b)
        
        # 効率計算
        efficiency_a = actual_speed_a_stats['mean'] / base_speed if base_speed > 0 else 0
        efficiency_b = actual_speed_b_stats['mean'] / base_speed if base_speed > 0 else 0
        avg_efficiency = (efficiency_a + efficiency_b) / 2
        
        # 制御効果評価
        control_utilization = correction_stats['mean'] / max(output_limits) if max(output_limits) > 0 else 0
        
        print(f"  📊 結果:")
        print(f"    平均速度差: {speed_stats['mean']:.1f} ±{speed_stats['std']:.1f}")
        print(f"    最大速度差: {speed_stats['max']:.0f}")
        print(f"    実測効率: A={efficiency_a:.3f}, B={efficiency_b:.3f}, 平均={avg_efficiency:.3f}")
        print(f"    制御使用率: {control_utilization:.3f} (1.0=制御限界)")
        print()
        
        return {
            'speed_diff_stats': speed_stats,
            'correction_stats': correction_stats,
            'efficiency': avg_efficiency,
            'control_utilization': control_utilization,
            'base_speed': base_speed,
            'pid_params': {'Kp': kp, 'Ki': ki, 'Kd': kd, 'limits': output_limits}
        }
    
    def compare_universal_settings(self):
        """汎用PID設定の比較テスト"""
        print("=" * 80)
        print("🔍 汎用PID設定探索 (70/98両対応)")
        print("=" * 80)
        
        # 汎用設定候補
        universal_configs = [
            # 超安全設定
            {'name': '超安全', 'kp': 0.3, 'ki': 0, 'kd': 0.3, 'limits': (-2, 2)},
            {'name': '安全', 'kp': 0.5, 'ki': 0, 'kd': 0.5, 'limits': (-3, 3)},
            
            # バランス設定
            {'name': 'バランス1', 'kp': 1.0, 'ki': 0, 'kd': 0.5, 'limits': (-4, 4)},
            {'name': 'バランス2', 'kp': 0.8, 'ki': 0, 'kd': 1.0, 'limits': (-4, 4)},
            {'name': 'バランス3', 'kp': 1.2, 'ki': 0, 'kd': 0.8, 'limits': (-5, 5)},
            
            # 応答性重視
            {'name': '応答1', 'kp': 1.5, 'ki': 0.1, 'kd': 1.0, 'limits': (-6, 6)},
            {'name': '応答2', 'kp': 2.0, 'ki': 0, 'kd': 1.5, 'limits': (-6, 6)},
            
            # 現在設定（参考）
            {'name': '現在設定', 'kp': 5.0, 'ki': 0, 'kd': 5.0, 'limits': (-8, 8)},
        ]
        
        speed_settings = [70, 98]
        all_results = {}
        
        input("車体を浮上状態にセットし、Enterキーを押してテスト開始...")
        
        for speed in speed_settings:
            print(f"\n=== HIGH_SPEED_BASE = {speed} ===")
            speed_results = {}
            
            for config in universal_configs:
                print(f"--- {config['name']} (速度{speed}) ---")
                
                result = self.test_universal_pid(
                    kp=config['kp'],
                    ki=config['ki'],
                    kd=config['kd'],
                    output_limits=config['limits'],
                    base_speed=speed,
                    duration=3.0
                )
                
                speed_results[config['name']] = result
                time.sleep(1)  # モーター安定化
            
            all_results[speed] = speed_results
        
        return all_results
    
    def analyze_universal_results(self, results: Dict):
        """汎用性分析"""
        print("\n" + "=" * 80)
        print("📊 汎用PID設定分析結果")
        print("=" * 80)
        
        # 各設定の汎用性を評価
        config_scores = {}
        
        for config_name in results[70].keys():
            # 70と98での性能を統合評価
            result_70 = results[70][config_name]
            result_98 = results[98][config_name]
            
            # 評価指標
            speed_diff_70 = result_70['speed_diff_stats']['mean']
            speed_diff_98 = result_98['speed_diff_stats']['mean']
            efficiency_70 = result_70['efficiency']
            efficiency_98 = result_98['efficiency']
            control_util_70 = result_70['control_utilization']
            control_util_98 = result_98['control_utilization']
            
            # 汎用性スコア計算
            # 1. 速度差の一貫性（小さく、かつ70/98で近い値）
            avg_speed_diff = (speed_diff_70 + speed_diff_98) / 2
            speed_diff_consistency = 1.0 / (1.0 + abs(speed_diff_70 - speed_diff_98))
            
            # 2. 効率の一貫性（高く、かつ70/98で近い値）
            avg_efficiency = (efficiency_70 + efficiency_98) / 2
            efficiency_consistency = 1.0 / (1.0 + abs(efficiency_70 - efficiency_98))
            
            # 3. 制御使用率のバランス（0.3-0.7が理想）
            avg_control_util = (control_util_70 + control_util_98) / 2
            control_balance = 1.0 - abs(avg_control_util - 0.5)  # 0.5が理想
            
            # 総合スコア
            universal_score = (
                (1.0 / avg_speed_diff) * 0.3 +  # 速度差性能
                speed_diff_consistency * 0.2 +   # 速度差一貫性
                avg_efficiency * 0.3 +           # 効率性能
                efficiency_consistency * 0.1 +   # 効率一貫性
                control_balance * 0.1             # 制御バランス
            )
            
            config_scores[config_name] = {
                'universal_score': universal_score,
                'avg_speed_diff': avg_speed_diff,
                'avg_efficiency': avg_efficiency,
                'avg_control_util': avg_control_util,
                'speed_diff_70': speed_diff_70,
                'speed_diff_98': speed_diff_98,
                'efficiency_70': efficiency_70,
                'efficiency_98': efficiency_98
            }
        
        # スコア順でソート
        sorted_configs = sorted(config_scores.items(), 
                               key=lambda x: x[1]['universal_score'], 
                               reverse=True)
        
        print("🏆 汎用PID設定ランキング:")
        for i, (name, scores) in enumerate(sorted_configs, 1):
            result_70 = results[70][name]
            result_98 = results[98][name]
            params = result_70['pid_params']
            
            print(f"{i}位: {name} (汎用スコア: {scores['universal_score']:.3f})")
            print(f"    設定: Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']}, limits={params['limits']}")
            print(f"    速度差: 70={scores['speed_diff_70']:.1f}, 98={scores['speed_diff_98']:.1f} (平均{scores['avg_speed_diff']:.1f})")
            print(f"    効率: 70={scores['efficiency_70']:.3f}, 98={scores['efficiency_98']:.3f} (平均{scores['avg_efficiency']:.3f})")
            print(f"    制御使用率: {scores['avg_control_util']:.3f}")
            print()
        
        # 推奨設定
        if sorted_configs:
            best_name, best_scores = sorted_configs[0]
            best_params = results[70][best_name]['pid_params']
            
            print("💡 推奨: 汎用PID設定")
            print(f"    pid.Kp = {best_params['Kp']}")
            print(f"    pid.Ki = {best_params['Ki']}")
            print(f"    pid.Kd = {best_params['Kd']}")
            print(f"    pid.output_limits = {best_params['limits']}")
            print()
            print("🎯 特徴:")
            print(f"  ✅ HIGH_SPEED_BASE 70/98 両対応")
            print(f"  ✅ 速度差一貫性: {best_scores['avg_speed_diff']:.1f}")
            print(f"  ✅ 効率一貫性: {best_scores['avg_efficiency']:.3f}")
            print(f"  ✅ 制御バランス: {best_scores['avg_control_util']:.3f}")
        
        return sorted_configs
    
    def run_universal_test(self):
        """汎用PIDテスト実行"""
        print("=" * 80)
        print("🚗 HIGH_SPEED_AVOID 汎用PID設定最適化テスト")
        print("=" * 80)
        print("目標: HIGH_SPEED_BASE 70/98 両方で最適動作する設定")
        print()
        
        # 比較テスト実行
        results = self.compare_universal_settings()
        
        # 結果分析
        ranking = self.analyze_universal_results(results)
        
        print(f"\n📅 テスト完了: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
        
        return results, ranking

if __name__ == "__main__":
    tester = UniversalPIDTester()
    
    print("汎用PID設定探索プログラム")
    print("HIGH_SPEED_BASE 70/98 両対応の最適設定を発見")
    print()
    
    # 汎用テスト実行
    results, ranking = tester.run_universal_test()
