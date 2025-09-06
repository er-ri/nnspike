#!/usr/bin/env python3
"""
最低限PID設定テスト - ライントレース安全性を考慮した極小制御

ベースライン性能に近い最低限のPID制御を見つけるテスト
"""

import time
import math
import sys
import os
from typing import List, Dict, Tuple

# パスを追加してパッケージをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE
from nnspike.utils import PIDController

class MinimalPIDTester:
    """最低限PID制御テストクラス"""
    
    def __init__(self):
        self.et = ETRobot()
        self.base_speed = HIGH_SPEED_BASE
        
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
    
    def test_minimal_pid(self, kp: float, ki: float, kd: float, 
                        output_limits: Tuple[float, float], 
                        simulated_line_error: float = 0,
                        duration: float = 5.0) -> Dict:
        """最低限PID制御テスト（ライン偏差シミュレーション付き）"""
        print(f"🔬 最低限PID Kp={kp}, Ki={ki}, Kd={kd}, limits={output_limits}")
        print(f"   ライン偏差シミュレーション: {simulated_line_error}")
        
        # PID制御器を設定
        pid = PIDController(
            Kp=kp, Ki=ki, Kd=kd,
            setpoint=0,
            output_limits=output_limits
        )
        
        speed_diffs = []
        corrections = []
        positions_a = []
        positions_b = []
        
        # 相対位置をリセット
        self.et.set_motor_relative_position(0, 0)
        
        start_time = time.time()
        initial_pos_a = None
        initial_pos_b = None
        
        while time.time() - start_time < duration:
            status = self.et.get_spike_status()
            
            pos_a = status.motors["A"].relative_position
            pos_b = status.motors["B"].relative_position
            speed_a = abs(status.motors["A"].speed or 0)
            speed_b = abs(status.motors["B"].speed or 0)
            
            if initial_pos_a is None and pos_a is not None and pos_b is not None:
                initial_pos_a = pos_a
                initial_pos_b = pos_b
            
            if pos_a is not None and pos_b is not None and initial_pos_a is not None:
                # 実際の位置偏差 + シミュレートされたライン偏差
                rel_pos_a = pos_a - initial_pos_a
                rel_pos_b = pos_b - initial_pos_b
                position_error = rel_pos_a - rel_pos_b + simulated_line_error
                
                # PID制御で補正
                correction = pid.update(position_error)
                corrections.append(abs(correction))
                
                # 補正を適用した速度設定
                left_speed = int(max(0, min(255, self.base_speed - correction)))
                right_speed = int(max(0, min(255, self.base_speed + correction)))
                
                self.et.set_motor_forward_speed(left_speed, right_speed)
                
                # データ記録
                positions_a.append(abs(rel_pos_a))
                positions_b.append(abs(rel_pos_b))
                
                if speed_a > 0 and speed_b > 0:
                    speed_diff = abs(speed_a - speed_b)
                    speed_diffs.append(speed_diff)
            
            time.sleep(0.05)  # 20Hz
        
        self.et.brake()
        
        # 評価指標計算
        final_pos_diff = 0
        if positions_a and positions_b:
            final_pos_a = positions_a[-1]
            final_pos_b = positions_b[-1]
            final_pos_diff = abs(final_pos_a - final_pos_b)
        
        speed_stats = self.calculate_stats(speed_diffs)
        correction_stats = self.calculate_stats(corrections)
        
        # ベースライン性能との比較
        baseline_pos_diff = 1  # 前回のベースライン結果
        baseline_speed_diff = 2.3
        
        pos_degradation = final_pos_diff / baseline_pos_diff if baseline_pos_diff > 0 else float('inf')
        speed_degradation = speed_stats['mean'] / baseline_speed_diff if baseline_speed_diff > 0 else float('inf')
        
        print(f"  📊 結果:")
        print(f"    平均速度差: {speed_stats['mean']:.1f} (ベースライン比: {speed_degradation:.1f}x)")
        print(f"    最終位置偏差: {final_pos_diff:.0f} (ベースライン比: {pos_degradation:.0f}x)")
        print(f"    平均PID補正: {correction_stats['mean']:.2f}")
        print(f"    制御の害: 位置×{pos_degradation:.0f}, 速度×{speed_degradation:.1f}")
        print()
        
        return {
            'speed_diff_stats': speed_stats,
            'correction_stats': correction_stats,
            'final_position_diff': final_pos_diff,
            'position_degradation': pos_degradation,
            'speed_degradation': speed_degradation,
            'total_degradation': (pos_degradation + speed_degradation) / 2,
            'pid_params': {'Kp': kp, 'Ki': ki, 'Kd': kd, 'limits': output_limits},
            'simulated_error': simulated_line_error
        }
    
    def find_optimal_minimal_pid(self):
        """最適な最低限PID設定を探索"""
        print("=" * 80)
        print("🔍 最低限PID設定探索")
        print("=" * 80)
        print("目標: ベースライン性能に最も近い最小限の制御")
        print()
        
        # 極小設定候補
        minimal_configs = [
            # 超極小制御
            {'name': '超極小1', 'kp': 0.5, 'ki': 0, 'kd': 0.5, 'limits': (-2, 2)},
            {'name': '超極小2', 'kp': 0.3, 'ki': 0, 'kd': 0.3, 'limits': (-1.5, 1.5)},
            {'name': '超極小3', 'kp': 0.1, 'ki': 0, 'kd': 0.1, 'limits': (-1, 1)},
            
            # 微調整版
            {'name': '微調整1', 'kp': 1, 'ki': 0, 'kd': 0.5, 'limits': (-3, 3)},
            {'name': '微調整2', 'kp': 0.8, 'ki': 0, 'kd': 0.2, 'limits': (-2.5, 2.5)},
            {'name': '微調整3', 'kp': 0.5, 'ki': 0, 'kd': 1, 'limits': (-2, 2)},
            
            # 安全マージン版
            {'name': '安全1', 'kp': 1.5, 'ki': 0, 'kd': 1, 'limits': (-4, 4)},
            {'name': '安全2', 'kp': 1, 'ki': 0.05, 'kd': 0.5, 'limits': (-3, 3)},
        ]
        
        results = []
        
        input("車体をセットし、Enterキーを押してテスト開始...")
        
        for config in minimal_configs:
            print(f"--- {config['name']} ---")
            
            # 通常状態テスト
            result = self.test_minimal_pid(
                kp=config['kp'],
                ki=config['ki'],
                kd=config['kd'],
                output_limits=config['limits'],
                simulated_line_error=0,  # 偏差なし
                duration=3.0
            )
            
            results.append({
                'config': config,
                'normal_result': result,
                'total_degradation': result['total_degradation']
            })
            
            time.sleep(1)  # モーター安定化
        
        # 結果分析
        results.sort(key=lambda x: x['total_degradation'])
        
        print("🏆 最低限PID設定ランキング (ベースライン性能に近い順):")
        for i, result in enumerate(results, 1):
            config = result['config']
            normal = result['normal_result']
            
            print(f"{i}位: {config['name']} (劣化度: {result['total_degradation']:.2f})")
            print(f"    Kp={config['kp']}, Ki={config['ki']}, Kd={config['kd']}")
            print(f"    制御幅={config['limits']}")
            print(f"    位置劣化×{normal['position_degradation']:.1f}, 速度劣化×{normal['speed_degradation']:.1f}")
            print()
        
        # 推奨設定
        if results:
            best = results[0]
            best_config = best['config']
            
            print("💡 推奨: 最低限PID設定")
            print(f"    pid.Kp = {best_config['kp']}")
            print(f"    pid.Ki = {best_config['ki']}")
            print(f"    pid.Kd = {best_config['kd']}")
            print(f"    pid.output_limits = {best_config['limits']}")
            print()
            print("🎯 特徴:")
            print("  ✅ ベースライン性能に最も近い")
            print("  ✅ ライン偏差時の最小限制御")
            print("  ✅ 自然な直進性能を最大限保持")
        
        return results
    
    def test_line_deviation_response(self, kp: float, ki: float, kd: float, 
                                   output_limits: Tuple[float, float]):
        """ライン偏差時の応答性テスト"""
        print(f"\n🚨 ライン偏差応答テスト")
        print(f"設定: Kp={kp}, Ki={ki}, Kd={kd}, limits={output_limits}")
        
        # 様々な偏差レベルでテスト
        deviation_levels = [5, 10, 20, 50]  # ピクセル偏差を想定
        
        for deviation in deviation_levels:
            print(f"\n--- ライン偏差 {deviation}px シミュレーション ---")
            result = self.test_minimal_pid(
                kp=kp, ki=ki, kd=kd,
                output_limits=output_limits,
                simulated_line_error=deviation,
                duration=2.0
            )
            
            correction_avg = result['correction_stats']['mean']
            print(f"    平均PID補正: {correction_avg:.2f}")
            print(f"    補正効果: {correction_avg/deviation:.3f} (理想値: 近い値)")
            
            time.sleep(1)

if __name__ == "__main__":
    tester = MinimalPIDTester()
    
    print("最低限PID設定探索プログラム")
    print("ライントレース安全性を考慮した極小制御の最適化")
    print()
    
    # 最適な最低限PID設定を探索
    results = tester.find_optimal_minimal_pid()
    
    if results:
        # 最優秀設定でライン偏差応答テスト
        best_config = results[0]['config']
        tester.test_line_deviation_response(
            kp=best_config['kp'],
            ki=best_config['ki'],
            kd=best_config['kd'],
            output_limits=best_config['limits']
        )
