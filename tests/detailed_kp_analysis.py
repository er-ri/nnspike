#!/usr/bin/env python3
"""
Kp=0.5～5.0詳細解析PIDテスト
効率と制御力のバランスから最適解を導出
"""

import sys
import os
import time
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class DetailedKpAnalysisTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_with_control_analysis(self, kp, kd, limits, base_speed, duration=1.5):
        """制御力とバランス分析付きPIDテスト"""
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
        
        print(f"⚡ Kp={kp}, Kd={kd}, limits={limits}, 速度={base_speed} ", end="")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                
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
        
        print(f"→ 効率{efficiency_score:.3f}, 制御力{control_score:.3f}, バランス{balance_score:.3f}")
        
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
            'balance_score': balance_score
        }
    
    def run_detailed_kp_analysis(self):
        """Kp=0.5～5.0詳細解析実行"""
        print("🔍 Kp=0.5～5.0 詳細バランス解析")
        print("=" * 50)
        print("🎯 効率と制御力のトレードオフ分析")
        print("⚖️ 最適バランスポイントを発見")
        
        # 詳細Kp範囲
        kd_values = [0, 0.3]  # Kd=0(効率重視) vs Kd=0.3(制御重視)
        kp_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # 10段階
        limits_values = [(-2, 2), (-3, 3), (-4, 4)]  # 3種類
        speeds = [70, 98]  # 両速度
        
        total_tests = len(kd_values) * len(kp_values) * len(limits_values) * len(speeds)
        print(f"📊 詳細テスト数: {total_tests}")
        print(f"   Kp: {kp_values}")
        print(f"   Kd: {kd_values}")
        print(f"   limits: {limits_values}")
        print(f"⏱️ 推定時間: {total_tests * 1.5 / 60:.1f}分")
        
        if input("車輪浮かせて詳細バランス解析開始？ (y/N): ").lower() != 'y':
            return
        
        print(f"\n🚀 Kp=0.5～5.0 効率×制御力バランス解析開始！")
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
                        
                        result = self.test_pid_with_control_analysis(
                            kp=kp, kd=kd, limits=limits, base_speed=speed
                        )
                        if result:
                            all_results.append(result)
                        
                        time.sleep(0.1)
        
        # 詳細分析
        self.analyze_kp_balance_results(all_results)
    
    def analyze_kp_balance_results(self, results):
        """Kp詳細バランス分析"""
        print("\n" + "=" * 80)
        print("🏆 Kp=0.5～5.0 効率×制御力バランス解析結果")
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
                    'limit_usage': []
                }
            kp_stats[kp]['efficiency'].append(result['efficiency_score'])
            kp_stats[kp]['control'].append(result['control_score'])
            kp_stats[kp]['balance'].append(result['balance_score'])
            kp_stats[kp]['limit_usage'].append(result['limit_usage'])
        
        print("\n📊 Kp値別性能統計:")
        print("┌──────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ Kp値 │ 効率平均 │ 制御平均 │ バランス │ 使用率   │ 総合評価 │")
        print("├──────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
        
        kp_rankings = []
        for kp in sorted(kp_stats.keys()):
            stats = kp_stats[kp]
            avg_eff = sum(stats['efficiency']) / len(stats['efficiency'])
            avg_ctrl = sum(stats['control']) / len(stats['control'])
            avg_bal = sum(stats['balance']) / len(stats['balance'])
            avg_usage = sum(stats['limit_usage']) / len(stats['limit_usage'])
            
            # 総合評価（バランススコアを重視）
            total_score = avg_bal
            
            print(f"│ {kp:4.1f} │ {avg_eff:8.3f} │ {avg_ctrl:8.3f} │ {avg_bal:8.3f} │ {avg_usage:8.3f} │ {total_score:8.3f} │")
            
            kp_rankings.append({
                'kp': kp,
                'efficiency': avg_eff,
                'control': avg_ctrl,
                'balance': avg_bal,
                'usage': avg_usage,
                'total': total_score
            })
        
        print("└──────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
        
        # ランキング
        kp_rankings.sort(key=lambda x: x['total'], reverse=True)
        
        print(f"\n🏅 Kp値総合ランキング（バランス重視）:")
        for i, rank in enumerate(kp_rankings[:8], 1):
            if rank['efficiency'] > 0.7 and rank['control'] > 0.2:
                badge = "🏆"
            elif rank['efficiency'] > 0.6:
                badge = "🥈"
            elif rank['control'] > 0.3:
                badge = "🥉"
            else:
                badge = "📊"
                
            print(f"{badge} {i}位: Kp={rank['kp']} (バランス{rank['total']:.3f})")
            print(f"       効率{rank['efficiency']:.3f} + 制御力{rank['control']:.3f} = 実用性評価")
        
        # 最適解分析
        winner = kp_rankings[0]
        runner_up = kp_rankings[1] if len(kp_rankings) > 1 else None
        
        print(f"\n🎯 最適解詳細分析:")
        print(f"🏆 1位: Kp={winner['kp']}")
        print(f"   効率: {winner['efficiency']:.3f} ({'高効率' if winner['efficiency'] > 0.7 else '中効率' if winner['efficiency'] > 0.5 else '低効率'})")
        print(f"   制御力: {winner['control']:.3f} ({'強力' if winner['control'] > 0.4 else '十分' if winner['control'] > 0.2 else '弱い'})")
        print(f"   バランス: {winner['balance']:.3f}")
        
        if runner_up:
            print(f"🥈 2位: Kp={runner_up['kp']} (バランス{runner_up['balance']:.3f})")
        
        # 推奨設定決定
        print(f"\n✅ 科学的根拠に基づく最終推奨:")
        
        # 効率重視 vs 制御重視の判定
        if winner['efficiency'] > 0.8 and winner['control'] < 0.3:
            recommendation_type = "効率重視型"
            kd_rec = 0
        elif winner['control'] > 0.4:
            recommendation_type = "制御重視型"
            kd_rec = 0.3
        else:
            recommendation_type = "バランス型"
            kd_rec = 0.3
        
        print(f"🎯 推奨設定 ({recommendation_type}):")
        print(f"   pid.Kp = {winner['kp']}")
        print(f"   pid.Kd = {kd_rec}")
        print(f"   pid.output_limits = (-4, 4)")
        
        print(f"\n💡 選択理由:")
        if winner['kp'] <= 1.0:
            print(f"  • Kp={winner['kp']}: 高速域の自然安定性を最大活用")
            print(f"  • 効率{winner['efficiency']:.3f}: エネルギー効率を重視")
        elif winner['kp'] <= 2.0:
            print(f"  • Kp={winner['kp']}: 効率と制御力の理想的バランス")
            print(f"  • 実用性を重視した設定")
        else:
            print(f"  • Kp={winner['kp']}: 強力な制御で確実な応答")
            print(f"  • 制御力{winner['control']:.3f}: 外乱に対する高い対応力")

def main():
    tester = DetailedKpAnalysisTest()
    try:
        tester.run_detailed_kp_analysis()
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
