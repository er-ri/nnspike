#!/usr/bin/env python3
"""
output_limits最適化テスト
Kp=1.0固定でoutput_limitsを細かく変えて最適値を科学的に検証
高速安定性効率と制御安定性の両立を目指す
"""

import sys
import os
import time
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class OutputLimitsOptimizationTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_output_limit(self, kp, kd, limit_value, base_speed, duration=5.0):
        """特定のoutput_limit値でのテスト"""
        limits = (-limit_value, limit_value)
        name = f"limits=±{limit_value}"
        
        print(f"\n=== {name} (速度{base_speed}) テスト開始 ===")
        print(f"設定: Kp={kp}, Kd={kd}, limits={limits}")
        print("車体を安全な場所にセットして準備...")
        input("準備完了したらEnterを押してください...")
        
        # PID設定
        pid = PIDController(
            Kp=kp,
            Ki=0,
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        # データ収集用
        positions = []
        speeds = []
        corrections = []
        limit_hits = 0  # 制限に達した回数
        
        print(f"🚗 {name} (速度{base_speed}) 5秒テスト開始！")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 微小な外乱を追加
                elapsed = time.time() - start_time
                if elapsed > 1:  # 1秒後から外乱開始
                    disturbance = random.uniform(-0.03, 0.03)  # より大きな外乱でテスト
                else:
                    disturbance = 0
                
                # PID制御
                steering_correction = pid.update(disturbance)
                
                # 制限値到達チェック
                if abs(steering_correction) >= limit_value * 0.95:  # 95%以上で制限と見なす
                    limit_hits += 1
                
                # 速度設定
                left_speed = base_speed - steering_correction
                right_speed = base_speed + steering_correction
                
                # 速度制限
                left_speed = max(0, min(255, left_speed))
                right_speed = max(0, min(255, right_speed))
                
                # モーター制御
                self.et.set_motor_forward_speed(
                    left_speed=int(left_speed),
                    right_speed=int(right_speed)
                )
                
                # データ記録
                status = self.et.get_spike_status()
                if status and status.motors:
                    left_pos = status.motors["A"].relative_position
                    right_pos = status.motors["B"].relative_position
                    positions.append((left_pos, right_pos))
                    speeds.append((left_speed, right_speed))
                    corrections.append(abs(steering_correction))
                
                time.sleep(0.02)  # 50Hz
        
        except KeyboardInterrupt:
            print("テスト中断")
        finally:
            self.et.brake()
            time.sleep(0.5)
        
        # 結果分析
        result = self.analyze_limit_result(name, base_speed, limit_value, positions, speeds, corrections, limit_hits)
        return result
    
    def analyze_limit_result(self, name, base_speed, limit_value, positions, speeds, corrections, limit_hits):
        """output_limits特化の結果分析"""
        if not positions:
            print(f"❌ {name} (速度{base_speed}): データ不足")
            return None
        
        # 基本統計
        avg_correction = sum(corrections) / len(corrections) if corrections else 0
        max_correction = max(corrections) if corrections else 0
        
        # 速度効率（左右の差が小さいほど良い）
        speed_diffs = []
        for left, right in speeds:
            speed_diffs.append(abs(left - right))
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs) if speed_diffs else 0
        
        # 制限使用率
        limit_usage_rate = limit_hits / len(corrections) if corrections else 0
        
        # 制御効率（制御量当たりの安定性）
        control_efficiency = 1.0 / (avg_correction + 0.1) if avg_correction > 0 else 10.0
        
        # 位置安定性
        if len(positions) > 1:
            left_positions = [p[0] for p in positions]
            right_positions = [p[1] for p in positions]
            left_stability = max(left_positions) - min(left_positions)
            right_stability = max(right_positions) - min(right_positions)
            position_stability = (left_stability + right_stability) / 2
        else:
            position_stability = 0
        
        # スピード効率スコア（速度差が小さく、制御量も適度）
        speed_efficiency = 1.0 / (1.0 + avg_speed_diff * 0.1 + avg_correction * 0.5)
        
        # 制御安定性スコア（外乱に対する応答性）
        control_stability = 1.0 / (1.0 + avg_correction + avg_speed_diff * 0.01)
        
        # 制限適正性（制限を使い切らず、かつ不足もしない）
        if limit_usage_rate > 0.8:  # 80%以上使用 = 制限不足
            limit_adequacy = 0.3
        elif limit_usage_rate < 0.1:  # 10%未満使用 = 制限過多
            limit_adequacy = 0.5
        else:  # 適度な使用
            limit_adequacy = 1.0
        
        # 総合効率スコア（高速安定性効率と制御安定性の両立）
        balanced_score = (speed_efficiency * 0.4 + control_stability * 0.4 + limit_adequacy * 0.2)
        
        print(f"\n📊 {name} (速度{base_speed}) 詳細結果:")
        print(f"   平均制御量: {avg_correction:.3f}")
        print(f"   最大制御量: {max_correction:.3f}")
        print(f"   速度差平均: {avg_speed_diff:.2f}")
        print(f"   制限使用率: {limit_usage_rate:.1%}")
        print(f"   制御効率: {control_efficiency:.3f}")
        print(f"   スピード効率: {speed_efficiency:.3f}")
        print(f"   制御安定性: {control_stability:.3f}")
        print(f"   制限適正性: {limit_adequacy:.3f}")
        print(f"   ⭐総合効率: {balanced_score:.3f}")
        
        return {
            'name': name,
            'base_speed': base_speed,
            'limit_value': limit_value,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'avg_speed_diff': avg_speed_diff,
            'limit_usage_rate': limit_usage_rate,
            'control_efficiency': control_efficiency,
            'speed_efficiency': speed_efficiency,
            'control_stability': control_stability,
            'limit_adequacy': limit_adequacy,
            'balanced_score': balanced_score,
            'position_stability': position_stability
        }
    
    def run_comprehensive_limits_test(self):
        """包括的なoutput_limits最適化テスト"""
        print("🔬 output_limits最適化テスト")
        print("=" * 60)
        print("⚠️ Kp=1.0, Kd=1.0固定でlimits値を細かく検証")
        print("⚠️ 高速安定性効率と制御安定性の両立を目指します")
        
        # 固定PID値
        kp = 1.0
        kd = 1.0
        
        # テストするlimits値（細かく刻む）
        limit_values = [
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
            5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0
        ]
        
        speeds = [70, 98]  # HIGH_SPEED_AVOID対応速度
        
        all_results = []
        
        for speed in speeds:
            print(f"\n🚀 速度{speed}でのlimits最適化テスト")
            for limit_value in limit_values:
                result = self.test_output_limit(
                    kp=kp,
                    kd=kd,
                    limit_value=limit_value,
                    base_speed=speed,
                    duration=5.0
                )
                if result:
                    all_results.append(result)
                
                print(f"limits=±{limit_value} (速度{speed}) 完了")
                time.sleep(1)  # 短い休憩
        
        # 最終分析
        self.final_limits_analysis(all_results)
    
    def final_limits_analysis(self, results):
        """output_limits最適化の最終分析"""
        print("\n" + "=" * 80)
        print("🏆 output_limits最適化最終結果")
        print("=" * 80)
        
        # 速度別に整理
        results_70 = [r for r in results if r['base_speed'] == 70]
        results_98 = [r for r in results if r['base_speed'] == 98]
        
        print("📊 速度70での詳細分析:")
        sorted_70 = sorted(results_70, key=lambda x: x['balanced_score'], reverse=True)
        for i, result in enumerate(sorted_70[:5], 1):  # トップ5
            print(f"{i}位: limits=±{result['limit_value']}")
            print(f"     総合効率: {result['balanced_score']:.3f}")
            print(f"     スピード効率: {result['speed_efficiency']:.3f}")
            print(f"     制御安定性: {result['control_stability']:.3f}")
            print(f"     制限使用率: {result['limit_usage_rate']:.1%}")
            print()
        
        print("📊 速度98での詳細分析:")
        sorted_98 = sorted(results_98, key=lambda x: x['balanced_score'], reverse=True)
        for i, result in enumerate(sorted_98[:5], 1):  # トップ5
            print(f"{i}位: limits=±{result['limit_value']}")
            print(f"     総合効率: {result['balanced_score']:.3f}")
            print(f"     スピード効率: {result['speed_efficiency']:.3f}")
            print(f"     制御安定性: {result['control_stability']:.3f}")
            print(f"     制限使用率: {result['limit_usage_rate']:.1%}")
            print()
        
        # 両速度での総合評価
        print("🎯 両速度総合評価:")
        limit_combined_scores = {}
        
        for result in results:
            limit_val = result['limit_value']
            if limit_val not in limit_combined_scores:
                limit_combined_scores[limit_val] = []
            limit_combined_scores[limit_val].append(result['balanced_score'])
        
        # 両速度のデータがある設定のみ評価
        final_rankings = []
        for limit_val, scores in limit_combined_scores.items():
            if len(scores) == 2:  # 両速度のデータ
                avg_score = sum(scores) / len(scores)
                consistency = 1.0 - abs(scores[0] - scores[1]) / max(scores)
                universal_score = avg_score * (0.7 + 0.3 * consistency)
                final_rankings.append((limit_val, universal_score, avg_score, consistency, scores))
        
        final_rankings.sort(key=lambda x: x[1], reverse=True)
        
        print("🏅 最終ランキング（両速度総合）:")
        for i, (limit_val, universal_score, avg_score, consistency, scores) in enumerate(final_rankings[:10], 1):
            print(f"{i}位: limits=±{limit_val}")
            print(f"     総合スコア: {universal_score:.3f}")
            print(f"     平均効率: {avg_score:.3f}")
            print(f"     一貫性: {consistency:.3f}")
            print(f"     速度別: 70={scores[0]:.3f}, 98={scores[1]:.3f}")
            print()
        
        # 最終推奨
        if final_rankings:
            winner_limit = final_rankings[0][0]
            winner_score = final_rankings[0][1]
            
            print(f"🥇 推奨output_limits: ±{winner_limit}")
            print(f"   総合スコア: {winner_score:.3f}")
            print(f"   高速安定性効率と制御安定性の最適バランス")
            
            # 詳細な推奨理由
            winner_results = [r for r in results if r['limit_value'] == winner_limit]
            if winner_results:
                avg_usage = sum(r['limit_usage_rate'] for r in winner_results) / len(winner_results)
                avg_speed_eff = sum(r['speed_efficiency'] for r in winner_results) / len(winner_results)
                avg_control_stab = sum(r['control_stability'] for r in winner_results) / len(winner_results)
                
                print(f"\n📋 推奨理由:")
                print(f"   制限使用率: {avg_usage:.1%} (適度な使用)")
                print(f"   スピード効率: {avg_speed_eff:.3f}")
                print(f"   制御安定性: {avg_control_stab:.3f}")
                print(f"   両速度での一貫性が高く、効率と安定性を両立")
                
                print(f"\n✅ 最終結論:")
                print(f"   output_limits=(-{winner_limit}, {winner_limit}) が最適")
                print(f"   ただし、これはKp=1.0, Kd=1.0固定での結果")
                print(f"   真の最適解にはKp、Kdも含めた包括的テストが必要")

def main():
    tester = OutputLimitsOptimizationTest()
    try:
        tester.run_comprehensive_limits_test()
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
