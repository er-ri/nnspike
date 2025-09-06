#!/usr/bin/env python3
"""
Extended Motor Smoothing Test - 詳細車体浮上テスト

異なるスムージング係数での詳細テストを実施
結果は保存せず、直接コンソールに出力
"""

import time
import math
import sys
import os
from typing import List, Dict

# パスを追加してパッケージをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

class ExtendedMotorSmoothingTest:
    """拡張モータースムージングテスト"""
    
    def __init__(self):
        self.et = ETRobot()
        self.base_speed = HIGH_SPEED_BASE  # 98
        
    def calculate_stats(self, data: List[float]) -> Dict[str, float]:
        """基本統計計算"""
        if not data:
            return {'mean': 0, 'max': 0, 'std': 0, 'min': 0}
        
        mean = sum(data) / len(data)
        max_val = max(data)
        min_val = min(data)
        
        if len(data) > 1:
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            std = variance ** 0.5
        else:
            std = 0
            
        return {
            'mean': mean,
            'max': max_val,
            'min': min_val,
            'std': std
        }
    
    def test_smoothing_factor(self, factor: float, base_speed: int = 98, duration: float = 3.0) -> Dict:
        """HIGH_SPEED_AVOID PID制限(±8)を前提としたスムージングテスト"""
        print(f"  スムージングテスト: factor={factor:.1f}, base={base_speed} (PID±8制限)")
        
        # スムージング変数
        current_left = 0.0
        current_right = 0.0
        
        # HIGH_SPEED_AVOID実際のPID補正パターン (output_limits=(-8, 8))
        test_sequence = [
            (base_speed, base_speed),           # 直進 (PID補正=0)
            (base_speed-8, base_speed+8),       # 最大右補正 (PID=-8, +8)
            (base_speed-4, base_speed+4),       # 中程度右補正 (PID=-4, +4)
            (base_speed+6, base_speed-6),       # 左補正 (PID=+6, -6)
            (base_speed+8, base_speed-8),       # 最大左補正 (PID=+8, -8)
            (base_speed-2, base_speed+2),       # 軽微補正 (PID=-2, +2)
            (base_speed, base_speed)            # 直進復帰 (PID=0)
        ]
        
        hold_time = duration / len(test_sequence)
        
        speed_diffs = []
        motor_a_speeds = []
        motor_b_speeds = []
        command_diffs = []  # 指令値差も記録
        
        for left_target, right_target in test_sequence:
            start_time = time.time()
            
            # 指令値の差を記録（実際のPID補正量）
            command_diff = abs(left_target - right_target)
            
            while time.time() - start_time < hold_time:
                # スムージング適用
                current_left += (left_target - current_left) * factor
                current_right += (right_target - current_right) * factor
                
                # モーター制御
                self.et.set_motor_forward_speed(int(current_left), int(current_right))
                
                # データ収集
                status = self.et.get_spike_status()
                motor_a_speed = status.motors["A"].speed or 0
                motor_b_speed = status.motors["B"].speed or 0
                
                speed_diff = abs(motor_a_speed - motor_b_speed)
                
                speed_diffs.append(speed_diff)
                motor_a_speeds.append(abs(motor_a_speed))
                motor_b_speeds.append(abs(motor_b_speed))
                command_diffs.append(command_diff)
                
                time.sleep(0.05)  # 20Hz
        
        self.et.brake()
        time.sleep(0.5)
        
        # 統計計算
        return {
            'factor': factor,
            'base_speed': base_speed,
            'speed_diff_stats': self.calculate_stats(speed_diffs),
            'motor_a_stats': self.calculate_stats(motor_a_speeds),
            'motor_b_stats': self.calculate_stats(motor_b_speeds),
            'command_diff_stats': self.calculate_stats(command_diffs),
            'pid_limits': '±8',
            'samples': len(speed_diffs)
        }
    
    def test_base_speed_stability(self, base_speed: int, duration: float = 4.0) -> Dict:
        """指定ベース速度での安定性テスト"""
        print(f"  ベース速度テスト: {base_speed}")
        
        speed_diffs = []
        motor_a_speeds = []
        motor_b_speeds = []
        
        # 固定速度での安定性テスト
        self.et.set_motor_forward_speed(base_speed, base_speed)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            status = self.et.get_spike_status()
            motor_a_speed = status.motors["A"].speed or 0
            motor_b_speed = status.motors["B"].speed or 0
            
            speed_diff = abs(motor_a_speed - motor_b_speed)
            
            speed_diffs.append(speed_diff)
            motor_a_speeds.append(abs(motor_a_speed))
            motor_b_speeds.append(abs(motor_b_speed))
            
            time.sleep(0.02)  # 50Hz
        
        self.et.brake()
        time.sleep(0.5)
        
        return {
            'base_speed': base_speed,
            'speed_diff_stats': self.calculate_stats(speed_diffs),
            'motor_a_stats': self.calculate_stats(motor_a_speeds),
            'motor_b_stats': self.calculate_stats(motor_b_speeds),
            'efficiency_a': self.calculate_stats(motor_a_speeds)['mean'] / base_speed if base_speed > 0 else 0,
            'efficiency_b': self.calculate_stats(motor_b_speeds)['mean'] / base_speed if base_speed > 0 else 0,
            'samples': len(speed_diffs)
        }
    
    def test_pid_simulation(self, duration: float = 4.0) -> Dict:
        """PID制御シミュレーションテスト"""
        print(f"  PIDシミュレーションテスト")
        
        speed_diffs = []
        corrections = []
        
        base_speed = self.base_speed
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # 疑似的なPID補正をシミュレート
            # サイン波で左右差を作る
            correction = int(15 * math.sin(elapsed * 2))  # ±15の補正
            
            left_speed = base_speed - correction
            right_speed = base_speed + correction
            
            self.et.set_motor_forward_speed(left_speed, right_speed)
            
            status = self.et.get_spike_status()
            motor_a_speed = status.motors["A"].speed or 0
            motor_b_speed = status.motors["B"].speed or 0
            
            speed_diff = abs(motor_a_speed - motor_b_speed)
            speed_diffs.append(speed_diff)
            corrections.append(abs(correction))
            
            time.sleep(0.02)  # 50Hz
        
        self.et.brake()
        
        return {
            'speed_diff_stats': self.calculate_stats(speed_diffs),
            'correction_stats': self.calculate_stats(corrections),
            'samples': len(speed_diffs)
        }
    
    def test_battery_simulation(self, duration: float = 3.0) -> Dict:
        """バッテリー電圧変動シミュレーションテスト"""
        print(f"  バッテリー電圧変動シミュレーション")
        
        speed_diffs = []
        voltage_factors = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # 電圧変動をシミュレート (100% -> 85% -> 100%)
            voltage_factor = 1.0 - 0.15 * abs(math.sin(elapsed))
            voltage_factors.append(voltage_factor)
            
            # 電圧に応じた速度調整
            adjusted_speed = int(self.base_speed * voltage_factor)
            
            self.et.set_motor_forward_speed(adjusted_speed, adjusted_speed)
            
            status = self.et.get_spike_status()
            motor_a_speed = status.motors["A"].speed or 0
            motor_b_speed = status.motors["B"].speed or 0
            
            speed_diff = abs(motor_a_speed - motor_b_speed)
            speed_diffs.append(speed_diff)
            
            time.sleep(0.05)  # 20Hz
        
        self.et.brake()
        
        return {
            'speed_diff_stats': self.calculate_stats(speed_diffs),
            'voltage_stats': self.calculate_stats(voltage_factors),
            'samples': len(speed_diffs)
        }
    
    def run_extended_tests(self):
        """HIGH_SPEED_AVOID PID制限(±8)に準拠した実用テスト"""
        print("=== HIGH_SPEED_AVOID PID制限準拠テスト ===")
        print(f"第一候補: 98 vs 次案候補: 70前後")
        print(f"PID制限: output_limits=(-8, 8)")
        print(f"実際の速度範囲: 98±8=90-106 / 70±8=62-78")
        print("車体を浮上させた状態で実行してください")
        
        input("準備ができたらEnterキーを押してください...")
        
        results = {}
        
        # 1. ベース速度安定性比較（実用範囲）
        print("\n1. ベース速度安定性比較")
        base_speeds = [70, 75, 80, 85, 90, 95, 98]
        
        for base_speed in base_speeds:
            results[f'base_{base_speed}'] = self.test_base_speed_stability(base_speed)
        
        # 2. 98でのスムージング効果（PID±8制限準拠）
        print("\n2. 98でのスムージング効果（PID±8制限）")
        practical_factors = [0.2, 0.3, 0.4]
        
        for factor in practical_factors:
            results[f'smooth_98_{factor:.1f}'] = self.test_smoothing_factor(factor, 98)
        
        # 3. 70でのスムージング効果（PID±8制限準拠）
        print("\n3. 70でのスムージング効果（PID±8制限）")
        for factor in practical_factors:
            results[f'smooth_70_{factor:.1f}'] = self.test_smoothing_factor(factor, 70)
        
        # 結果表示
        self.display_practical_results(results)
        
        return results
    
    def display_practical_results(self, results: Dict):
        """実用的な結果表示"""
        print("\n" + "="*80)
        print("📊 実用範囲安定性テスト結果")
        print("="*80)
        
        # ベース速度安定性比較
        print("\n🎯 ベース速度安定性比較 (70-98):")
        print(f"{'速度':<6} {'平均差':<8} {'標準偏差':<10} {'効率A':<8} {'効率B':<8} {'サンプル数':<8}")
        print("-" * 60)
        
        base_results = [(k, v) for k, v in results.items() if k.startswith('base_')]
        base_results.sort(key=lambda x: x[1]['base_speed'])
        
        best_stability = None
        best_efficiency = None
        
        for test_name, data in base_results:
            base_speed = data['base_speed']
            stats = data['speed_diff_stats']
            eff_a = data['efficiency_a']
            eff_b = data['efficiency_b']
            
            # 安定性スコア（標準偏差が低いほど良い）
            stability_score = stats['std']
            
            # 効率スコア（1.0に近いほど良い）
            efficiency_score = abs(1.0 - (eff_a + eff_b) / 2)
            
            if best_stability is None or stability_score < best_stability[1]:
                best_stability = (base_speed, stability_score)
            
            if best_efficiency is None or efficiency_score < best_efficiency[1]:
                best_efficiency = (base_speed, efficiency_score)
            
            print(f"{base_speed:<6} {stats['mean']:<8.1f} {stats['std']:<10.1f} {eff_a:<8.3f} {eff_b:<8.3f} {data['samples']:<8}")
        
        # ベスト結果表示
        if best_stability:
            print(f"\n🏆 最安定速度: {best_stability[0]} (標準偏差: {best_stability[1]:.1f})")
        if best_efficiency:
            print(f"⚡ 最高効率速度: {best_efficiency[0]} (効率差: {best_efficiency[1]:.3f})")
        
        # スムージング効果比較
        print(f"\n🔧 スムージング効果比較:")
        print(f"{'設定':<12} {'係数':<6} {'平均差':<8} {'標準偏差':<10} {'改善効果':<10}")
        print("-" * 60)
        
        # 98ベースライン
        baseline_98 = next((v for k, v in results.items() if k == 'base_98'), None)
        baseline_70 = next((v for k, v in results.items() if k == 'base_70'), None)
        
        smooth_results = [(k, v) for k, v in results.items() if k.startswith('smooth_')]
        
        for test_name, data in smooth_results:
            base_speed = data['base_speed']
            factor = data['factor']
            stats = data['speed_diff_stats']
            
            # ベースラインとの比較
            baseline = baseline_98 if base_speed == 98 else baseline_70
            if baseline:
                improvement = baseline['speed_diff_stats']['std'] - stats['std']
                improvement_pct = (improvement / baseline['speed_diff_stats']['std']) * 100
                improvement_str = f"{improvement_pct:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{base_speed}+smooth{factor:<1.0f} {factor:<6.1f} {stats['mean']:<8.1f} {stats['std']:<10.1f} {improvement_str:<10}")
        
        # 結論と推奨事項
        print(f"\n📋 結論と推奨事項:")
        
        if baseline_98 and baseline_70:
            stability_98 = baseline_98['speed_diff_stats']['std']
            stability_70 = baseline_70['speed_diff_stats']['std']
            
            if stability_98 <= stability_70 * 1.1:  # 10%以内の差なら98推奨
                print(f"✅ 第一候補98推奨: 安定性差は許容範囲内 ({stability_98:.1f} vs {stability_70:.1f})")
            else:
                print(f"⚠️  70が安定: 98は不安定 ({stability_98:.1f} vs {stability_70:.1f})")
                print(f"   次案として70を検討")
        
        # 最適スムージング
        best_smooth = min(smooth_results, key=lambda x: x[1]['speed_diff_stats']['std'])
        best_smooth_data = best_smooth[1]
        print(f"🎯 最適スムージング: 速度{best_smooth_data['base_speed']} + 係数{best_smooth_data['factor']:.1f}")
        print(f"   期待効果: 標準偏差 {best_smooth_data['speed_diff_stats']['std']:.1f}")
        
        print(f"\n💡 実装方針:")
        print(f"1. 第一候補: HIGH_SPEED_BASE=98")
        print(f"2. 次案: HIGH_SPEED_BASE=70（安定性重視）")
        print(f"3. スムージング: 係数{best_smooth_data['factor']:.1f}を適用")
        print(f"4. 適用箇所: PID制御後段（et.set_motor_forward_speed前）")
    
    def display_results(self, results: Dict):
        """結果表示"""
        print("\n" + "="*80)
        print("📊 拡張テスト結果")
        print("="*80)
        
        # スムージング係数比較
        print("\n🎯 スムージング係数比較:")
        print(f"{'係数':<6} {'平均差':<8} {'最大差':<8} {'標準偏差':<10} {'最小差':<8} {'サンプル数':<8}")
        print("-" * 60)
        
        smooth_results = [(k, v) for k, v in results.items() if k.startswith('smooth_')]
        smooth_results.sort(key=lambda x: x[1]['factor'])
        
        for test_name, data in smooth_results:
            stats = data['speed_diff_stats']
            factor = data['factor']
            print(f"{factor:<6.1f} {stats['mean']:<8.1f} {stats['max']:<8.0f} {stats['std']:<10.1f} {stats['min']:<8.0f} {data['samples']:<8}")
        
        # 最適係数の特定
        best_factor = min(smooth_results, key=lambda x: x[1]['speed_diff_stats']['std'])
        print(f"\n🏆 最適係数: {best_factor[1]['factor']:.1f} (標準偏差: {best_factor[1]['speed_diff_stats']['std']:.1f})")
        
        # PIDシミュレーション結果
        if 'pid_simulation' in results:
            print(f"\n⚙️  PIDシミュレーション結果:")
            pid_stats = results['pid_simulation']['speed_diff_stats']
            corr_stats = results['pid_simulation']['correction_stats']
            print(f"   速度差: 平均={pid_stats['mean']:.1f}, 標準偏差={pid_stats['std']:.1f}")
            print(f"   補正値: 平均={corr_stats['mean']:.1f}, 最大={corr_stats['max']:.1f}")
        
        # バッテリーシミュレーション結果
        if 'battery_simulation' in results:
            print(f"\n🔋 バッテリーシミュレーション結果:")
            bat_stats = results['battery_simulation']['speed_diff_stats']
            vol_stats = results['battery_simulation']['voltage_stats']
            print(f"   速度差: 平均={bat_stats['mean']:.1f}, 標準偏差={bat_stats['std']:.1f}")
            print(f"   電圧係数: 最小={vol_stats['min']:.3f}, 平均={vol_stats['mean']:.3f}")
        
        # 実装推奨事項
        print(f"\n📋 実装推奨事項:")
        print(f"1. 最適スムージング係数: {best_factor[1]['factor']:.1f}")
        print(f"2. 期待改善効果: 標準偏差 {best_factor[1]['speed_diff_stats']['std']:.1f}")
        print(f"3. 適用条件: HIGH_SPEED_AVOID モード")
        print(f"4. 実装箇所: PID制御の後段（et.set_motor_forward_speed の直前）")

def main():
    tester = ExtendedMotorSmoothingTest()
    
    try:
        tester.run_extended_tests()
        print("\n✅ 拡張テスト完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
