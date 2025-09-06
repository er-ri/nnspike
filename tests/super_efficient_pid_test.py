#!/usr/bin/env python3
"""
Kp=0.5～5.0詳細解析PIDテスト
効率と制御力のバランスから最適解を導出
Raspberry Pi専用（Windowsでは動作しません）
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
    
    def run_detailed_kp_analysis(self, test_mode="full"):
        """Kp=0.5～5.0詳細解析実行"""
        print("🔍 Kp=0.5～5.0 詳細バランス解析")
        print("=" * 50)
        print("🎯 効率と制御力のトレードオフ分析")
        print("⚖️ 最適バランスポイントを発見")
        
        # テストモード別設定
        if test_mode == "quick":
            # クイックテスト：1分程度
            kd_values = [0.3]  # 既存設定のみ
            kp_values = [0.8, 1.0, 1.5, 2.0]  # 重要4点
            limits_values = [(-4, 4)]  # 現在設定のみ
            speeds = [98]  # 高速のみ
            print("🚀 クイックテスト（約1分）")
        elif test_mode == "medium":
            # 中程度テスト：2-3分程度  
            kd_values = [0, 0.3, 0.5]
            kp_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]  # 6点
            limits_values = [(-2, 2), (-4, 4)]
            speeds = [98]
            print("⚖️ 中程度テスト（約2-3分）")
        else:  # full
            # フルテスト：4-5分（オリジナル）
            kd_values = [0, 0.3, 0.5, 1.0]  # Kd=0も検証、0.3は従来値
            kp_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]  # Kp=0.5～5.0を細かく
            limits_values = [(-2, 2), (-4, 4), (-8, 8)]  # 制御範囲も幅広く
            speeds = [70, 98]  # 2種類
            print("🎯 フルテスト（約4-5分）")
        
        total_tests = len(kd_values) * len(kp_values) * len(limits_values) * len(speeds)
        estimated_time = total_tests * 1.5 / 60  # 1.5秒/テスト
        print(f"📊 テスト数: {total_tests}")
        print(f"⏱️ 推定時間: {estimated_time:.1f}分")
        print(f"   Kd: {kd_values}")
        print(f"   Kp: {kp_values}")  
        print(f"   制御範囲: {limits_values}")
        print(f"   速度: {speeds}")
        print(f"   1テスト: 1.5秒")
        
        if input("車輪浮かせてKp細かい解析テスト開始？ (y/N): ").lower() != 'y':
            return
        
        print(f"\n🚀 Kp=0.5～5.0細かい解析テスト開始！")
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
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_airborne(self, kp, kd, limits, base_speed, duration=1.0):
        """超高速PIDテスト（1秒）- 制御力とバランス重視分析"""
        pid = PIDController(
            Kp=kp,
            Ki=0,
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        corrections = []
        speed_diffs = []
        large_errors = []  # 大きなずれの制御能力測定
        
        print(f"⚡ Kp={kp}, Kd={kd}, limits={limits}, 速度={base_speed} ", end="")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                
                # 段階的外乱（制御力テスト）
                if elapsed < 0.2:
                    disturbance = 0  # 安定期
                elif elapsed < 0.5:
                    disturbance = random.uniform(-0.02, 0.02)  # 小外乱
                elif elapsed < 0.8:
                    disturbance = random.uniform(-0.05, 0.05)  # 中外乱
                else:
                    disturbance = random.uniform(-0.08, 0.08)  # 大外乱（制御力測定）
                
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
                
                # 大外乱時の制御力評価
                if elapsed > 0.8 and abs(disturbance) > 0.05:
                    large_errors.append(abs(steering_correction))
                
                time.sleep(0.02)
                
        finally:
            self.et.brake()
            time.sleep(0.1)
        
        # 制御力とバランス分析
        if not corrections:
            return None
        
        avg_correction = sum(corrections) / len(corrections)
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
        
        # 制御力評価（大外乱への対応）
        control_power = sum(large_errors) / len(large_errors) if large_errors else 0
        max_correction = max(corrections) if corrections else 0
        
        # 制御範囲評価（output_limitsに対する使用率）
        limit_usage = max_correction / limits[1] if limits[1] > 0 else 0
        
        # バランススコア（効率 vs 制御力）
        efficiency_factor = 1.0 / (1.0 + avg_correction * 0.5)
        control_factor = min(1.0, control_power / 2.0)  # 制御力正規化
        balance_score = (efficiency_factor * 0.6 + control_factor * 0.4)
        
        print(f"→ 制御{avg_correction:.2f}, 効率{efficiency_factor:.3f}, 制御力{control_power:.2f}, バランス{balance_score:.3f}")
        
        return {
            'kp': kp,
            'kd': kd,
            'limits': limits,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'avg_speed_diff': avg_speed_diff,
            'efficiency_factor': efficiency_factor,
            'control_power': control_power,
            'max_correction': max_correction,
            'limit_usage': limit_usage,
            'balance_score': balance_score
        }
    
    def run_detailed_kp_analysis(self):
        """Kp=0.5～5.0詳細解析実行"""
        print("⚡ Kp=0.5～5.0詳細解析PIDテスト")
        print("=" * 50)
        print("🔍 Kp=0.5～5.0を細かく解析")
        print("🎯 効率と制御力のバランスから最適解を導出")
        print("🚨 Raspberry Pi専用（Windowsでは動作しません）")
        
        # Kp=0.5～5.0の詳細解析
        kd_values = [0, 0.3, 0.5]  # 主要Kd値
        kp_values = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # Kp詳細解析
        limits_values = [(-2, 2), (-4, 4), (-6, 6), (-8, 8)]  # 制御範囲詳細
        speeds = [70, 98]  # 2種類
        
        total_tests = len(kd_values) * len(kp_values) * len(limits_values) * len(speeds)
        print(f"📊 Kp詳細解析テスト数: {total_tests}")
        print(f"   Kd: {kd_values}")
        print(f"   Kp: {kp_values} ← 詳細解析")
        print(f"   limits: {limits_values}")
        print(f"⏱️ 推定時間: {total_tests * 1.5 / 60:.1f}分")
        
        if input("🚨 Raspberry Piで詳細Kp解析開始？ (y/N): ").lower() != 'y':
            return
        
        print(f"\n🚀 Kp=0.5～5.0詳細解析開始！")
        all_results = []
        test_count = 0
        
        for speed in speeds:
            print(f"\n📈 速度{speed}:")
            for kd in kd_values:
                for kp in kp_values:
                    for limits in limits_values:
                        test_count += 1
                        print(f"[{test_count:2d}/{total_tests}] ", end="")
                        
                        result = self.test_pid_with_control_analysis(kp=kp, kd=kd, limits=limits, base_speed=speed)
                        if result:
                            all_results.append(result)
                        
                        time.sleep(0.1)  # 最小休憩
        
        # 結果分析
        self.analyze_detailed_kp_results(all_results)
    
    def analyze_detailed_kp_results(self, results):
        """Kp=0.5～5.0詳細分析"""
        print("\n" + "=" * 80)
        print("🏆 Kp=0.5～5.0 詳細解析結果")
        print("=" * 80)
        
        if not results:
            print("❌ 結果なし")
            return
        
        # Kp別効率と制御力分析
        kp_analysis = {}
        for result in results:
            kp = result['kp']
            if kp not in kp_analysis:
                kp_analysis[kp] = {
                    'efficiency': [],
                    'control_power': [],
                    'balance': [],
                    'limit_usage': []
                }
            kp_analysis[kp]['efficiency'].append(result['efficiency_factor'])
            kp_analysis[kp]['control_power'].append(result['control_power'])
            kp_analysis[kp]['balance'].append(result['balance_score'])
            kp_analysis[kp]['limit_usage'].append(result['limit_usage'])
        
        print("\n📊 Kp値別性能分析:")
        print("Kp値 | 効率平均 | 制御力平均 | バランス平均 | 制御使用率 | 総合評価")
        print("-" * 70)
        
        kp_rankings = []
        for kp in sorted(kp_analysis.keys()):
            data = kp_analysis[kp]
            avg_eff = sum(data['efficiency']) / len(data['efficiency'])
            avg_ctrl = sum(data['control_power']) / len(data['control_power'])
            avg_bal = sum(data['balance']) / len(data['balance'])
            avg_usage = sum(data['limit_usage']) / len(data['limit_usage'])
            
            # 総合評価（効率40%, 制御力30%, バランス30%）
            total_score = avg_eff * 0.4 + avg_ctrl * 0.3 + avg_bal * 0.3
            
            print(f"{kp:4.1f} | {avg_eff:8.3f} | {avg_ctrl:10.3f} | {avg_bal:12.3f} | {avg_usage:10.3f} | {total_score:8.3f}")
            
            kp_rankings.append({
                'kp': kp,
                'efficiency': avg_eff,
                'control_power': avg_ctrl,
                'balance': avg_bal,
                'limit_usage': avg_usage,
                'total_score': total_score
            })
        
        # ランキング
        kp_rankings.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"\n🏅 Kp値総合ランキング:")
        for i, rank in enumerate(kp_rankings[:5], 1):
            print(f"{i}位: Kp={rank['kp']} (総合{rank['total_score']:.3f})")
            print(f"     効率{rank['efficiency']:.3f}, 制御力{rank['control_power']:.3f}, バランス{rank['balance']:.3f}")
        
        # 最適解の決定理由
        winner = kp_rankings[0]
        print(f"\n🎯 最適解: Kp={winner['kp']}")
        print(f"選定理由:")
        
        if winner['efficiency'] > 0.8:
            print(f"  ✅ 高効率: {winner['efficiency']:.3f}")
        else:
            print(f"  ⚠️ 効率やや低: {winner['efficiency']:.3f}")
            
        if winner['control_power'] > 0.3:
            print(f"  ✅ 十分な制御力: {winner['control_power']:.3f}")
        else:
            print(f"  ⚠️ 制御力不足: {winner['control_power']:.3f}")
            
        if winner['balance'] > 0.7:
            print(f"  ✅ 良好なバランス: {winner['balance']:.3f}")
        else:
            print(f"  ⚠️ バランス課題: {winner['balance']:.3f}")
        
        # Kd=0 vs Kd=0.3 比較
        print(f"\n🔬 Kd=0 vs Kd=0.3 比較:")
        kd_comparison = {}
        for result in results:
            kd = result['kd']
            if kd not in kd_comparison:
                kd_comparison[kd] = []
            kd_comparison[kd].append(result['balance_score'])
        
        for kd in sorted(kd_comparison.keys()):
            scores = kd_comparison[kd]
            avg_score = sum(scores) / len(scores)
            print(f"  Kd={kd}: 平均バランス{avg_score:.3f}")
        
        # 最終推奨設定
        print(f"\n✅ 最終推奨設定 (効率と制御力のバランス):")
        print(f"   Kp={winner['kp']}")
        print(f"   Kd=0 (効率重視) または Kd=0.3 (制御安定重視)")
        print(f"   output_limits=(-4, 4) (制御範囲とのバランス)")
        
        # 理論的説明
        print(f"\n💡 理論的根拠:")
        if winner['kp'] < 1.0:
            print(f"  • Kp={winner['kp']}: 高速域の自然安定性を活用")
            print(f"  • 過制御を避け、車体特性を尊重")
        elif winner['kp'] > 3.0:
            print(f"  • Kp={winner['kp']}: 強力な制御で外乱に対応")
            print(f"  • 制御精度を重視した設定")
        else:
            print(f"  • Kp={winner['kp']}: 効率と制御力の最適バランス")
            print(f"  • 実用的な制御範囲での安定動作")
        """超効率化結果分析"""
        print("\n" + "=" * 60)
        print("🏆 制御力とバランス重視テスト結果")
        print("=" * 60)
        
        if not results:
            print("❌ 結果なし")
            return
        
        # バランススコア順ソート
        sorted_by_balance = sorted(results, key=lambda x: x['balance_score'], reverse=True)
        
        print("\n🏅 バランススコアランキング（TOP10）:")
        for i, result in enumerate(sorted_by_balance[:10], 1):
            print(f"{i:2d}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}, 速度={result['base_speed']}")
            print(f"      バランス: {result['balance_score']:.3f}, 効率: {result['efficiency_factor']:.3f}, 制御力: {result['control_power']:.2f}")
            print(f"      制御範囲使用率: {result['limit_usage']:.1%}, 最大制御: {result['max_correction']:.2f}")
        
        # 効率のみでのランキング
        sorted_by_efficiency = sorted(results, key=lambda x: x['efficiency_factor'], reverse=True)
        print("\n⚡ 効率のみランキング（TOP5）:")
        for i, result in enumerate(sorted_by_efficiency[:5], 1):
            print(f"{i}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}, 速度={result['base_speed']}")
            print(f"     効率: {result['efficiency_factor']:.3f}, 制御力: {result['control_power']:.2f}")
        
        # 制御力のみでのランキング
        sorted_by_control = sorted(results, key=lambda x: x['control_power'], reverse=True)
        print("\n🛡️ 制御力のみランキング（TOP5）:")
        for i, result in enumerate(sorted_by_control[:5], 1):
            print(f"{i}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}, 速度={result['base_speed']}")
            print(f"     制御力: {result['control_power']:.2f}, 効率: {result['efficiency_factor']:.3f}")
        
        # Kp詳細分析（最重要）
        print(f"\n🔍 Kp値詳細分析 (0.5～5.0):")
        kp_detailed = {}
        for result in results:
            kp = result['kp']
            if kp not in kp_detailed:
                kp_detailed[kp] = {
                    'balance': [], 'efficiency': [], 'control': [],
                    'max_control': [], 'limit_usage': []
                }
            kp_detailed[kp]['balance'].append(result['balance_score'])
            kp_detailed[kp]['efficiency'].append(result['efficiency_factor'])
            kp_detailed[kp]['control'].append(result['control_power'])
            kp_detailed[kp]['max_control'].append(result['max_correction'])
            kp_detailed[kp]['limit_usage'].append(result['limit_usage'])
        
        print("   Kp値別詳細性能:")
        for kp in sorted(kp_detailed.keys()):
            data = kp_detailed[kp]
            avg_balance = sum(data['balance']) / len(data['balance'])
            avg_efficiency = sum(data['efficiency']) / len(data['efficiency'])
            avg_control = sum(data['control']) / len(data['control'])
            avg_max_control = sum(data['max_control']) / len(data['max_control'])
            avg_limit_usage = sum(data['limit_usage']) / len(data['limit_usage'])
            
            print(f"   Kp={kp:3.1f}: バランス{avg_balance:.3f}, 効率{avg_efficiency:.3f}, 制御力{avg_control:.2f}")
            print(f"         最大制御{avg_max_control:.2f}, 制御範囲使用{avg_limit_usage:.1%}")
        
        # Kp最適解分析
        print(f"\n🎯 Kp最適解分析:")
        kp_scores = []
        for kp in sorted(kp_detailed.keys()):
            data = kp_detailed[kp]
            avg_balance = sum(data['balance']) / len(data['balance'])
            avg_efficiency = sum(data['efficiency']) / len(data['efficiency'])
            avg_control = sum(data['control']) / len(data['control'])
            
            # 総合評価（バランス重視）
            comprehensive_score = avg_balance * 0.6 + avg_efficiency * 0.3 + min(avg_control/3.0, 1.0) * 0.1
            
            kp_scores.append({
                'kp': kp,
                'comprehensive': comprehensive_score,
                'balance': avg_balance,
                'efficiency': avg_efficiency,
                'control': avg_control
            })
        
        kp_scores.sort(key=lambda x: x['comprehensive'], reverse=True)
        
        print("   Kp総合ランキング:")
        for i, score_data in enumerate(kp_scores[:5], 1):
            kp = score_data['kp']
            print(f"   {i}位: Kp={kp:3.1f} (総合{score_data['comprehensive']:.3f})")
            print(f"        バランス{score_data['balance']:.3f}, 効率{score_data['efficiency']:.3f}, 制御{score_data['control']:.2f}")
        
        # 最適Kp推奨理由
        optimal_kp = kp_scores[0]
        print(f"\n💡 最適Kp推奨: {optimal_kp['kp']}")
        print(f"   理由分析:")
        print(f"   • 総合スコア: {optimal_kp['comprehensive']:.3f} (最高)")
        print(f"   • バランス: {optimal_kp['balance']:.3f}")
        print(f"   • 効率: {optimal_kp['efficiency']:.3f}")
        print(f"   • 制御力: {optimal_kp['control']:.2f}")
        
        # Kp=0.5 vs Kp=5.0比較
        kp_05_data = next((x for x in kp_scores if x['kp'] == 0.5), None)
        kp_50_data = next((x for x in kp_scores if x['kp'] == 5.0), None)
        
        if kp_05_data and kp_50_data:
            print(f"\n⚖️ Kp=0.5 vs Kp=5.0 詳細比較:")
            print(f"   Kp=0.5: 総合{kp_05_data['comprehensive']:.3f}, 効率{kp_05_data['efficiency']:.3f}, 制御{kp_05_data['control']:.2f}")
            print(f"   Kp=5.0: 総合{kp_50_data['comprehensive']:.3f}, 効率{kp_50_data['efficiency']:.3f}, 制御{kp_50_data['control']:.2f}")
            
            if kp_05_data['comprehensive'] > kp_50_data['comprehensive']:
                winner = "Kp=0.5"
                reason = "効率重視で総合バランスが優秀"
            else:
                winner = "Kp=5.0"
                reason = "制御力重視で安定性が優秀"
            
            print(f"   勝者: {winner} ({reason})")

        # Kp別分析（制御力重視）
        print(f"\n📊 Kp値別分析:")
        kp_analysis = {}
        for result in results:
            kp = result['kp']
            if kp not in kp_analysis:
                kp_analysis[kp] = {'balance': [], 'efficiency': [], 'control': []}
            kp_analysis[kp]['balance'].append(result['balance_score'])
            kp_analysis[kp]['efficiency'].append(result['efficiency_factor'])
            kp_analysis[kp]['control'].append(result['control_power'])
        
        for kp in sorted(kp_analysis.keys()):
            data = kp_analysis[kp]
            avg_balance = sum(data['balance']) / len(data['balance'])
            avg_efficiency = sum(data['efficiency']) / len(data['efficiency'])
            avg_control = sum(data['control']) / len(data['control'])
            print(f"  Kp={kp}: バランス{avg_balance:.3f}, 効率{avg_efficiency:.3f}, 制御力{avg_control:.2f}")
        
        # Kd別分析
        print(f"\n📊 Kd値別分析:")
        kd_analysis = {}
        for result in results:
            kd = result['kd']
            if kd not in kd_analysis:
                kd_analysis[kd] = {'balance': [], 'efficiency': [], 'control': []}
            kd_analysis[kd]['balance'].append(result['balance_score'])
            kd_analysis[kd]['efficiency'].append(result['efficiency_factor'])
            kd_analysis[kd]['control'].append(result['control_power'])
        
        for kd in sorted(kd_analysis.keys()):
            data = kd_analysis[kd]
            avg_balance = sum(data['balance']) / len(data['balance'])
            avg_efficiency = sum(data['efficiency']) / len(data['efficiency'])
            avg_control = sum(data['control']) / len(data['control'])
            print(f"  Kd={kd}: バランス{avg_balance:.3f}, 効率{avg_efficiency:.3f}, 制御力{avg_control:.2f}")
        
        # 両速度対応分析
        print(f"\n🎯 両速度対応分析:")
        dual_analysis = {}
        for result in results:
            key = (result['kp'], result['kd'], result['limits'])
            if key not in dual_analysis:
                dual_analysis[key] = []
            dual_analysis[key].append(result)
        
        dual_candidates = []
        for key, speed_results in dual_analysis.items():
            if len(speed_results) == 2:  # 両速度
                avg_balance = sum(r['balance_score'] for r in speed_results) / 2
                avg_efficiency = sum(r['efficiency_factor'] for r in speed_results) / 2
                avg_control = sum(r['control_power'] for r in speed_results) / 2
                consistency = 1.0 - abs(speed_results[0]['balance_score'] - speed_results[1]['balance_score'])
                dual_score = avg_balance * (0.9 + 0.1 * consistency)
                
                dual_candidates.append({
                    'kp': key[0],
                    'kd': key[1],
                    'limits': key[2],
                    'dual_score': dual_score,
                    'avg_balance': avg_balance,
                    'avg_efficiency': avg_efficiency,
                    'avg_control': avg_control,
                    'consistency': consistency,
                    'results': speed_results
                })
        
        dual_candidates.sort(key=lambda x: x['dual_score'], reverse=True)
        
        print("🏅 両速度総合ランキング（バランス重視）:")
        for i, candidate in enumerate(dual_candidates[:5], 1):  # TOP5のみ
            print(f"{i}位: Kp={candidate['kp']}, Kd={candidate['kd']}, limits={candidate['limits']}")
            print(f"     総合スコア: {candidate['dual_score']:.3f}")
            print(f"     バランス: {candidate['avg_balance']:.3f}, 効率: {candidate['avg_efficiency']:.3f}, 制御力: {candidate['avg_control']:.2f}")
        
        # 最終推奨（バランス重視）
        if dual_candidates:
            winner = dual_candidates[0]
            print(f"\n🏆 最終推奨設定（バランス重視）:")
            print(f"   Kp={winner['kp']}, Kd={winner['kd']}, limits={winner['limits']}")
            print(f"   総合スコア: {winner['dual_score']:.3f}")
            print(f"   バランス: {winner['avg_balance']:.3f}, 効率: {winner['avg_efficiency']:.3f}, 制御力: {winner['avg_control']:.2f}")
            
            print(f"\n✅ 制御力重視テスト結論:")
            print(f"   HIGH_SPEED_AVOIDモード推奨設定:")
            print(f"   Kp={winner['kp']}, Kd={winner['kd']}, output_limits={winner['limits']}")
            
            # 効率最高との比較
            efficiency_best = sorted_by_efficiency[0]
            print(f"\n⚖️ バランス最優秀 vs 効率最優秀:")
            print(f"   バランス最優秀: Kp={winner['kp']}, Kd={winner['kd']} (バランス{winner['avg_balance']:.3f})")
            print(f"   効率最優秀: Kp={efficiency_best['kp']}, Kd={efficiency_best['kd']} (効率{efficiency_best['efficiency_factor']:.3f})")
            
            if winner['kp'] != efficiency_best['kp'] or winner['kd'] != efficiency_best['kd']:
                print(f"   ⚠️ 効率重視とバランス重視で推奨が異なります")
                print(f"   実走行では制御能力も重要です")
            
        else:
            # 単一最優秀
            best = sorted_by_balance[0]
            print(f"🏆 単一最優秀（バランス重視）:")
            print(f"   Kp={best['kp']}, Kd={best['kd']}, limits={best['limits']}")
            print(f"   バランススコア: {best['balance_score']:.3f}")
        
        if not results:
            print("❌ 結果なし")
            return
        
        # 効率順ソート
        sorted_results = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
        
        print("\n🥇 効率ランキング（TOP10）:")
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"{i:2d}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}, 速度={result['base_speed']}")
            print(f"      効率スコア: {result['efficiency_score']:.3f}")
            print(f"      制御量: {result['avg_correction']:.2f}, 速度差: {result['avg_speed_diff']:.1f}")
        
        # Kd別分析
        print(f"\n📊 Kd値別平均効率:")
        kd_analysis = {}
        for result in results:
            kd = result['kd']
            if kd not in kd_analysis:
                kd_analysis[kd] = []
            kd_analysis[kd].append(result['efficiency_score'])
        
        for kd in sorted(kd_analysis.keys()):
            scores = kd_analysis[kd]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"  Kd={kd}: 平均{avg_score:.3f}, 最高{max_score:.3f} (サンプル{len(scores)})")
        
        # 両速度対応分析
        print(f"\n🎯 両速度対応分析:")
        dual_analysis = {}
        for result in results:
            key = (result['kp'], result['kd'], result['limits'])
            if key not in dual_analysis:
                dual_analysis[key] = []
            dual_analysis[key].append(result)
        
        dual_candidates = []
        for key, speed_results in dual_analysis.items():
            if len(speed_results) == 2:  # 両速度
                avg_efficiency = sum(r['efficiency_score'] for r in speed_results) / 2
                consistency = 1.0 - abs(speed_results[0]['efficiency_score'] - speed_results[1]['efficiency_score']) / max(r['efficiency_score'] for r in speed_results)
                dual_score = avg_efficiency * (0.8 + 0.2 * consistency)
                
                dual_candidates.append({
                    'kp': key[0],
                    'kd': key[1],
                    'limits': key[2],
                    'dual_score': dual_score,
                    'avg_efficiency': avg_efficiency,
                    'consistency': consistency,
                    'results': speed_results
                })
        
        dual_candidates.sort(key=lambda x: x['dual_score'], reverse=True)
        
        print("🏅 両速度総合ランキング:")
        for i, candidate in enumerate(dual_candidates[:5], 1):  # TOP5のみ
            print(f"{i}位: Kp={candidate['kp']}, Kd={candidate['kd']}, limits={candidate['limits']}")
            print(f"     両速度スコア: {candidate['dual_score']:.3f}")
            print(f"     平均効率: {candidate['avg_efficiency']:.3f}, 一貫性: {candidate['consistency']:.3f}")
        
        # 最終推奨
        if dual_candidates:
            winner = dual_candidates[0]
            print(f"\n🏆 最終推奨設定:")
            print(f"   Kp={winner['kp']}, Kd={winner['kd']}, limits={winner['limits']}")
            print(f"   両速度スコア: {winner['dual_score']:.3f}")
            
            print(f"\n✅ 1分テスト結論:")
            print(f"   HIGH_SPEED_AVOIDモード最適設定:")
            print(f"   Kp={winner['kp']}, Kd={winner['kd']}, output_limits={winner['limits']}")
            
            # Kd=0の評価も表示
            kd_zero_results = [r for r in results if r['kd'] == 0]
            if kd_zero_results:
                best_kd_zero = max(kd_zero_results, key=lambda x: x['efficiency_score'])
                print(f"\n💡 Kd=0の最高性能:")
                print(f"   Kp={best_kd_zero['kp']}, Kd=0, limits={best_kd_zero['limits']}")
                print(f"   効率: {best_kd_zero['efficiency_score']:.3f} (推奨設定比較用)")
        else:
            # 単一最優秀
            best = sorted_results[0]
            print(f"🏆 単一最優秀:")
            print(f"   Kp={best['kp']}, Kd={best['kd']}, limits={best['limits']}")
            print(f"   効率スコア: {best['efficiency_score']:.3f}")

def main():
    """メイン関数 - テストモード選択付き"""
    import sys
    
    # コマンドライン引数でモード選択
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
    else:
        # インタラクティブ選択
        print("\n🔧 PIDテストモード選択:")
        print("1. quick  - クイックテスト（約1分、4テスト）")
        print("2. medium - 中程度テスト（約2-3分、36テスト）") 
        print("3. full   - フルテスト（約4-5分、192テスト）")
        
        choice = input("\nモードを選択してください [1/2/3]: ").strip()
        test_mode = {"1": "quick", "2": "medium", "3": "full"}.get(choice, "full")
    
    print(f"\n🚀 {test_mode}モードで実行します")
    
    tester = DetailedKpAnalysisTest()
    try:
        tester.run_detailed_kp_analysis(test_mode)
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
