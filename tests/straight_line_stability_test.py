#!/usr/bin/env python3
"""
直進安定性テスト - HIGH_SPEED_AVOIDモードのPID設定最適化

HIGH_SPEED_BASE=98での直進安定性を詳細分析し、
PID制御パラメータの最適化を行います。
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

class StraightLineStabilityTester:
    """直進安定性専用テストクラス"""
    
    def __init__(self):
        self.et = ETRobot()
        self.base_speed = HIGH_SPEED_BASE  # 98
        self.test_results = []
        
    def calculate_stats(self, data: List[float]) -> Dict:
        """統計計算（NumPy不使用）"""
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
    
    def test_straight_line_baseline(self, duration: float = 5.0) -> Dict:
        """ベースライン直進テスト（PID制御なし）"""
        print(f"🔬 ベースライン直進テスト ({duration}秒)")
        print("PID制御なし、純粋なモーター性能評価")
        
        speed_diffs = []
        positions_a = []
        positions_b = []
        
        # 相対位置をリセット
        self.et.set_motor_relative_position(0, 0)
        
        # 固定速度で直進
        self.et.set_motor_forward_speed(self.base_speed, self.base_speed)
        
        start_time = time.time()
        initial_pos_a = None
        initial_pos_b = None
        
        while time.time() - start_time < duration:
            status = self.et.get_spike_status()
            
            pos_a = status.motors["A"].relative_position
            pos_b = status.motors["B"].relative_position
            speed_a = abs(status.motors["A"].speed or 0)
            speed_b = abs(status.motors["B"].speed or 0)
            
            if initial_pos_a is None:
                initial_pos_a = pos_a
                initial_pos_b = pos_b
            
            # 累積位置偏差を記録
            if pos_a is not None and pos_b is not None:
                positions_a.append(abs(pos_a - initial_pos_a))
                positions_b.append(abs(pos_b - initial_pos_b))
                
                # 速度差を記録
                if speed_a > 0 and speed_b > 0:
                    speed_diff = abs(speed_a - speed_b)
                    speed_diffs.append(speed_diff)
            
            time.sleep(0.05)  # 20Hz
        
        self.et.brake()
        
        # 直進性評価（左右の累積位置差）
        final_pos_diff = 0
        if positions_a and positions_b:
            final_pos_a = positions_a[-1]
            final_pos_b = positions_b[-1]
            final_pos_diff = abs(final_pos_a - final_pos_b)
        
        speed_stats = self.calculate_stats(speed_diffs)
        
        print(f"  📊 結果:")
        print(f"    平均速度差: {speed_stats['mean']:.1f} ±{speed_stats['std']:.1f}")
        print(f"    最大速度差: {speed_stats['max']:.0f}")
        print(f"    最終位置偏差: {final_pos_diff:.0f} (小さいほど直進性良好)")
        print()
        
        return {
            'speed_diff_stats': speed_stats,
            'final_position_diff': final_pos_diff,
            'duration': duration,
            'samples': len(speed_diffs)
        }
    
    def test_pid_straight_line(self, kp: float, ki: float, kd: float, 
                              output_limits: Tuple[float, float], 
                              duration: float = 5.0) -> Dict:
        """PID制御付き直進テスト"""
        print(f"🎯 PID直進テスト Kp={kp}, Ki={ki}, Kd={kd}, limits={output_limits}")
        
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
                # 累積位置差を計算
                rel_pos_a = pos_a - initial_pos_a
                rel_pos_b = pos_b - initial_pos_b
                position_error = rel_pos_a - rel_pos_b  # 左右の位置偏差
                
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
        
        # PID効果評価
        control_effectiveness = correction_stats['mean'] / speed_stats['mean'] if speed_stats['mean'] > 0 else 0
        
        print(f"  📊 結果:")
        print(f"    平均速度差: {speed_stats['mean']:.1f} ±{speed_stats['std']:.1f}")
        print(f"    最大速度差: {speed_stats['max']:.0f}")
        print(f"    最終位置偏差: {final_pos_diff:.0f}")
        print(f"    平均PID補正: {correction_stats['mean']:.1f}")
        print(f"    制御効果: {control_effectiveness:.3f} (高いほど良好)")
        print()
        
        return {
            'speed_diff_stats': speed_stats,
            'correction_stats': correction_stats,
            'final_position_diff': final_pos_diff,
            'control_effectiveness': control_effectiveness,
            'pid_params': {'Kp': kp, 'Ki': ki, 'Kd': kd, 'limits': output_limits},
            'duration': duration,
            'samples': len(speed_diffs)
        }
    
    def compare_pid_settings(self, duration: float = 5.0) -> Dict:
        """複数のPID設定を比較"""
        print("🔬 PID設定比較テスト")
        print("=" * 60)
        
        # テスト設定
        test_configs = [
            # 現在の設定
            {'name': '現在設定', 'kp': 5, 'ki': 0, 'kd': 5, 'limits': (-8, 8)},
            
            # 直進特化設定
            {'name': '直進特化1', 'kp': 3, 'ki': 0.1, 'kd': 2, 'limits': (-6, 6)},
            {'name': '直進特化2', 'kp': 2, 'ki': 0, 'kd': 3, 'limits': (-5, 5)},
            {'name': '穏やか制御', 'kp': 1, 'ki': 0, 'kd': 1, 'limits': (-4, 4)},
            
            # 制御幅変更
            {'name': '制御幅拡大', 'kp': 5, 'ki': 0, 'kd': 5, 'limits': (-12, 12)},
        ]
        
        results = {}
        
        # ベースライン測定
        print("まずベースライン（PID制御なし）を測定...")
        baseline = self.test_straight_line_baseline(duration)
        results['baseline'] = baseline
        
        time.sleep(2)  # モーター安定化待ち
        
        # 各PID設定をテスト
        for config in test_configs:
            print(f"--- {config['name']} ---")
            result = self.test_pid_straight_line(
                kp=config['kp'],
                ki=config['ki'], 
                kd=config['kd'],
                output_limits=config['limits'],
                duration=duration
            )
            results[config['name']] = result
            
            time.sleep(2)  # モーター安定化待ち
        
        return results
    
    def analyze_results(self, results: Dict):
        """結果分析と推奨事項"""
        print("=" * 80)
        print("📊 直進安定性テスト結果分析")
        print("=" * 80)
        
        baseline = results.get('baseline', {})
        baseline_pos_diff = baseline.get('final_position_diff', float('inf'))
        baseline_speed_diff = baseline.get('speed_diff_stats', {}).get('mean', float('inf'))
        
        print(f"🔍 ベースライン性能:")
        print(f"  最終位置偏差: {baseline_pos_diff:.0f}")
        print(f"  平均速度差: {baseline_speed_diff:.1f}")
        print()
        
        # PID設定の性能ランキング
        pid_results = []
        
        for name, result in results.items():
            if name == 'baseline':
                continue
                
            pos_diff = result.get('final_position_diff', float('inf'))
            speed_diff = result.get('speed_diff_stats', {}).get('mean', float('inf'))
            control_eff = result.get('control_effectiveness', 0)
            
            # 総合スコア計算（位置偏差50%、速度差30%、制御効果20%）
            pos_score = max(0, 1 - pos_diff / max(baseline_pos_diff, 1))
            speed_score = max(0, 1 - speed_diff / max(baseline_speed_diff, 1))
            control_score = min(1, control_eff)
            
            total_score = pos_score * 0.5 + speed_score * 0.3 + control_score * 0.2
            
            pid_results.append({
                'name': name,
                'position_diff': pos_diff,
                'speed_diff': speed_diff,
                'control_effectiveness': control_eff,
                'total_score': total_score,
                'params': result.get('pid_params', {})
            })
        
        # スコア順でソート
        pid_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        print("🏆 PID設定ランキング (直進安定性):")
        for i, result in enumerate(pid_results, 1):
            params = result['params']
            print(f"{i}位: {result['name']} (スコア: {result['total_score']:.3f})")
            print(f"    Kp={params.get('Kp', 0)}, Ki={params.get('Ki', 0)}, Kd={params.get('Kd', 0)}")
            print(f"    制御幅={params.get('limits', (0, 0))}")
            print(f"    位置偏差={result['position_diff']:.0f}, 速度差={result['speed_diff']:.1f}")
            print()
        
        # 推奨事項
        if pid_results:
            best = pid_results[0]
            current = next((r for r in pid_results if r['name'] == '現在設定'), None)
            
            print("💡 推奨事項:")
            if current and best['total_score'] > current['total_score'] * 1.1:
                print(f"✅ 推奨: {best['name']} への変更")
                print(f"   改善効果: スコア {current['total_score']:.3f} → {best['total_score']:.3f}")
                best_params = best['params']
                print(f"   設定変更:")
                print(f"     pid.Kp = {best_params.get('Kp', 0)}")
                print(f"     pid.Ki = {best_params.get('Ki', 0)}")
                print(f"     pid.Kd = {best_params.get('Kd', 0)}")
                print(f"     pid.output_limits = {best_params.get('limits', (0, 0))}")
            else:
                print("✅ 現在設定が最適、変更不要")
        
        return pid_results
    
    def run_comprehensive_test(self, duration: float = 5.0):
        """包括的直進安定性テスト"""
        print("=" * 80)
        print("🚗 HIGH_SPEED_AVOID 直進安定性最適化テスト")
        print("=" * 80)
        print(f"📋 テスト条件:")
        print(f"  HIGH_SPEED_BASE: {self.base_speed}")
        print(f"  テスト時間: {duration}秒/設定")
        print(f"  評価項目: 位置偏差、速度差、制御効果")
        print()
        
        input("車体を直進可能な状態にセットし、Enterキーを押してください...")
        
        # 比較テスト実行
        results = self.compare_pid_settings(duration)
        
        # 結果分析
        ranking = self.analyze_results(results)
        
        print(f"\n📅 テスト完了: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
        
        return results, ranking

if __name__ == "__main__":
    tester = StraightLineStabilityTester()
    
    print("直進安定性テストプログラム")
    print("HIGH_SPEED_AVOIDモードのPID設定最適化")
    print()
    
    # 包括的テスト実行
    results, ranking = tester.run_comprehensive_test(duration=5.0)
