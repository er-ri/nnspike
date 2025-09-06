#!/usr/bin/env python3
"""
段階的PIDテスト - 時間調整可能
クイック/中程度/フルテストモード対応
"""

import sys
import os
import time
import random

# パスを追加してnnspike.etrobotをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from nnspike.unit import ETRobot
from simple_pid import PIDController

class StagedPIDTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_staged(self, kp, kd, limits, base_speed, duration=1.0):
        """段階的PIDテスト - パワー分析付き"""
        pid = PIDController(
            Kp=kp,
            Ki=0,
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        corrections = []
        speed_diffs = []
        large_disturbance_responses = []
        motor_powers = []  # パワー出力記録
        motor_speeds = []  # 実際の速度記録
        
        print(f"⚡ Kp={kp}, Kd={kd}, limits={limits} ", end="")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                
                # パワーと速度の記録
                status = self.et.get_spike_status()
                left_power = getattr(status.motors["A"], "power", 0) if hasattr(status.motors["A"], "power") else 0
                right_power = getattr(status.motors["B"], "power", 0) if hasattr(status.motors["B"], "power") else 0
                left_speed_actual = getattr(status.motors["A"], "speed", 0) if hasattr(status.motors["A"], "speed") else 0
                right_speed_actual = getattr(status.motors["B"], "speed", 0) if hasattr(status.motors["B"], "speed") else 0
                
                motor_powers.append((abs(left_power), abs(right_power)))
                motor_speeds.append((abs(left_speed_actual), abs(right_speed_actual)))
                
                # 段階的外乱（実際のライントレースを模擬）
                if elapsed < 0.3:
                    disturbance = 0  # 安定期
                elif elapsed < 0.8:
                    disturbance = random.uniform(-0.02, 0.02)  # 小外乱
                else:
                    # 大外乱（急カーブ・障害物回避を模擬）
                    disturbance = random.uniform(-0.08, 0.08)
                
                steering_correction = pid.update(disturbance)
                left_speed = base_speed - steering_correction
                right_speed = base_speed + steering_correction
                
                left_speed = max(0, min(255, left_speed))
                right_speed = max(0, min(255, right_speed))
                
                self.et.set_motor_forward_speed(
                    left_speed=int(left_speed),
                    right_speed=int(right_speed)
                )
                
                corrections.append(abs(steering_correction))
                speed_diffs.append(abs(left_speed - right_speed))
                
                # 大外乱時の制御応答を記録
                if elapsed > 0.8 and abs(disturbance) > 0.05:
                    large_disturbance_responses.append(abs(steering_correction))
                
                time.sleep(0.02)
                
        finally:
            self.et.brake()
            time.sleep(0.1)
        
        # 詳細分析
        if not corrections:
            return None
        
        avg_correction = sum(corrections) / len(corrections)
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
        max_correction = max(corrections)
        
        # パワー効率分析
        if motor_powers:
            avg_left_power = sum(p[0] for p in motor_powers) / len(motor_powers)
            avg_right_power = sum(p[1] for p in motor_powers) / len(motor_powers)
            max_left_power = max(p[0] for p in motor_powers)
            max_right_power = max(p[1] for p in motor_powers)
            avg_total_power = (avg_left_power + avg_right_power) / 2
            max_total_power = max(max_left_power, max_right_power)
        else:
            avg_total_power = max_total_power = 0
        
        # 速度効率分析
        if motor_speeds:
            avg_left_speed = sum(s[0] for s in motor_speeds) / len(motor_speeds)
            avg_right_speed = sum(s[1] for s in motor_speeds) / len(motor_speeds)
            avg_actual_speed = (avg_left_speed + avg_right_speed) / 2
        else:
            avg_actual_speed = 0
        
        # パワー効率：実際速度/パワー消費比
        power_efficiency = avg_actual_speed / max(1, avg_total_power)
        
        # 制御力評価
        control_power = (sum(large_disturbance_responses) / len(large_disturbance_responses)) if large_disturbance_responses else 0
        
        # 制御範囲使用率
        limit_usage = max_correction / limits[1] if limits[1] > 0 else 0
        
        # 効率評価（低制御ほど高効率）
        efficiency_score = 1.0 / (1.0 + avg_correction * 2.0)
        
        # 制御力評価（大外乱への対応力）
        control_score = min(1.0, control_power / 2.0)
        
        # バランススコア（効率50% + 制御力30% + 安定性20%）
        stability_score = 1.0 - min(1.0, avg_speed_diff / 10.0)
        balance_score = efficiency_score * 0.5 + control_score * 0.3 + stability_score * 0.2
        
        print(f"→ 効率{efficiency_score:.3f}, 制御{control_score:.3f}, バランス{balance_score:.3f}, パワー効率{power_efficiency:.2f}")
        
        return {
            'kp': kp,
            'kd': kd,
            'limits': limits,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'avg_speed_diff': avg_speed_diff,
            'max_correction': max_correction,
            'control_power': control_power,
            'limit_usage': limit_usage,
            'efficiency_score': efficiency_score,
            'control_score': control_score,
            'stability_score': stability_score,
            'balance_score': balance_score,
            'power_efficiency': power_efficiency,
            'avg_total_power': avg_total_power,
            'max_total_power': max_total_power,
            'avg_actual_speed': avg_actual_speed
        }
    
    def run_staged_test(self, test_mode="medium"):
        """段階的テスト実行"""
        print(f"🔧 段階的PIDテスト - {test_mode}モード")
        print("=" * 50)
        
        # テストモード別設定
        if test_mode == "quick":
            # クイックテスト：約1分
            kd_values = [0.3]  # 既存設定のみ
            kp_values = [0.8, 1.0, 1.5, 2.0]  # 重要4点
            limits_values = [(-4, 4)]  # 現在設定のみ
            speeds = [98]  # 高速のみ
            duration = 0.8  # 短縮
            print("🚀 クイックテスト（約1分）")
        elif test_mode == "medium":
            # 中程度テスト：約2-3分  
            kd_values = [0, 0.3, 0.5]
            kp_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]  # 6点
            limits_values = [(-2, 2), (-4, 4)]
            speeds = [98]
            duration = 1.0
            print("⚖️ 中程度テスト（約2-3分）")
        else:  # full
            # フルテスト：約4-5分
            kd_values = [0, 0.3, 0.5, 1.0]
            kp_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            limits_values = [(-2, 2), (-4, 4), (-8, 8)]
            speeds = [70, 98]
            duration = 1.5
            print("🎯 フルテスト（約4-5分）")
        
        total_tests = len(kd_values) * len(kp_values) * len(limits_values) * len(speeds)
        estimated_time = total_tests * duration / 60  # 分
        print(f"📊 テスト数: {total_tests}")
        print(f"⏱️ 推定時間: {estimated_time:.1f}分")
        print(f"   Kd: {kd_values}")
        print(f"   Kp: {kp_values}")
        print(f"   制御範囲: {limits_values}")
        print(f"   速度: {speeds}")
        print(f"   1テスト: {duration}秒")
        
        # 確認
        response = input(f"\n⚠️ {test_mode}テスト（{estimated_time:.1f}分）を開始しますか？ [y/N]: ")
        if response.lower() != 'y':
            print("❌ テスト中止")
            return
        
        print(f"\n🚀 {total_tests}テスト開始！")
        all_results = []
        test_count = 0
        
        for speed in speeds:
            print(f"\n📈 速度{speed}:")
            for kd in kd_values:
                print(f"  📐 Kd={kd}:")
                for kp in kp_values:
                    for limits in limits_values:
                        test_count += 1
                        print(f"    [{test_count:2d}/{total_tests}] ", end="")
                        
                        result = self.test_pid_staged(
                            kp=kp, kd=kd, limits=limits, base_speed=speed, duration=duration
                        )
                        if result:
                            all_results.append(result)
                        
                        time.sleep(0.1)
        
        # 詳細分析
        self.analyze_staged_results(all_results)
    
    def analyze_staged_results(self, results):
        """段階テスト結果分析"""
        print("\n" + "=" * 80)
        print("🏆 段階的PIDテスト結果分析（パワー効率含む）")
        print("=" * 80)
        
        if not results:
            print("❌ 結果なし")
            return
        
        # Kp値別統計
        kp_stats = {}
        for result in results:
            kp = result['kp']
            if kp not in kp_stats:
                kp_stats[kp] = {
                    'efficiency': [],
                    'control': [],
                    'balance': [],
                    'limit_usage': [],
                    'power_efficiency': [],
                    'avg_power': [],
                    'max_power': []
                }
            kp_stats[kp]['efficiency'].append(result['efficiency_score'])
            kp_stats[kp]['control'].append(result['control_score'])
            kp_stats[kp]['balance'].append(result['balance_score'])
            kp_stats[kp]['limit_usage'].append(result['limit_usage'])
            kp_stats[kp]['power_efficiency'].append(result['power_efficiency'])
            kp_stats[kp]['avg_power'].append(result['avg_total_power'])
            kp_stats[kp]['max_power'].append(result['max_total_power'])
        
        # 統計表示
        print("\n📊 Kp値別統計（パワー効率分析付き）:")
        print("┌──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ Kp値 │ 効率平均 │ 制御平均 │ バランス │ 使用率   │ パワー効率│ 平均電力 │ 総合評価 │")
        print("├──────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
        
        kp_rankings = []
        for kp in sorted(kp_stats.keys()):
            stats = kp_stats[kp]
            avg_eff = sum(stats['efficiency']) / len(stats['efficiency'])
            avg_ctrl = sum(stats['control']) / len(stats['control'])
            avg_bal = sum(stats['balance']) / len(stats['balance'])
            avg_usage = sum(stats['limit_usage']) / len(stats['limit_usage'])
            avg_power_eff = sum(stats['power_efficiency']) / len(stats['power_efficiency'])
            avg_power = sum(stats['avg_power']) / len(stats['avg_power'])
            
            # 総合評価（バランススコアを重視）
            total_score = avg_bal
            
            print(f"│ {kp:4.1f} │ {avg_eff:8.3f} │ {avg_ctrl:8.3f} │ {avg_bal:8.3f} │ {avg_usage:8.3f} │ {avg_power_eff:8.2f} │ {avg_power:8.1f} │ {total_score:8.3f} │")
            
            kp_rankings.append({
                'kp': kp,
                'efficiency': avg_eff,
                'control': avg_ctrl,
                'balance': avg_bal,
                'usage': avg_usage,
                'power_efficiency': avg_power_eff,
                'avg_power': avg_power,
                'total': total_score
            })
        
        print("└──────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
        
        # 最適解選出
        winner = max(kp_rankings, key=lambda x: x['total'])
        
        print(f"\n🎯 最適バランス解:")
        print(f"   🏆 Kp = {winner['kp']}")
        print(f"   📊 バランススコア: {winner['total']:.3f}")
        print(f"   ⚡ 効率: {winner['efficiency']:.3f}")
        print(f"   🎛️ 制御力: {winner['control']:.3f}")
        print(f"   🔋 パワー効率: {winner['power_efficiency']:.2f}")
        print(f"   ⚙️ 平均電力: {winner['avg_power']:.1f}")
        
        print(f"\n💡 推奨設定:")
        print(f"   HIGH_SPEED_AVOID: Kp={winner['kp']}, Kd=0.3, limits=(-4,4)")

def main():
    """メイン関数 - テストモード選択付き"""
    import sys
    
    # コマンドライン引数でモード選択
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
    else:
        # インタラクティブ選択
        print("\n🔧 段階的PIDテストモード選択:")
        print("1. quick  - クイックテスト（約1分、4テスト）")
        print("2. medium - 中程度テスト（約2-3分、36テスト）") 
        print("3. full   - フルテスト（約4-5分、192テスト）")
        
        choice = input("\nモードを選択してください [1/2/3]: ").strip()
        test_mode = {"1": "quick", "2": "medium", "3": "full"}.get(choice, "medium")
    
    print(f"\n🚀 {test_mode}モードで実行します")
    
    tester = StagedPIDTest()
    try:
        tester.run_staged_test(test_mode)
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
