#!/usr/bin/env python3
"""
包括的PIDパラメータ分析テスト
Kp、Kd、output_limitsの全組み合わせを網羅的にテスト
高速安定性効率と制御安定性の最適バランスを科学的に発見
"""

import sys
import os
import time
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class ComprehensivePIDLimitsAnalysis:
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_combination(self, kp, kd, limits, base_speed, duration=5.0):
        """特定のPID+limits組み合わせをテスト"""
        name = f"Kp={kp}_Kd={kd}_lim={limits}"
        print(f"\n=== {name} (速度{base_speed}) テスト開始 ===")
        print("車体を安全な場所にセットして準備...")
        input("準備完了したらEnterを押してください...")
        
        # PID設定
        pid = PIDController(
            Kp=kp,
            Ki=0,  # Kiは常に0で固定
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        # データ収集用
        corrections = []
        speed_diffs = []
        positions = []
        
        print(f"🚗 {name} (速度{base_speed}) 5秒テスト開始！")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 微小な外乱を追加
                elapsed = time.time() - start_time
                if elapsed > 1:  # 1秒後から外乱開始
                    disturbance = random.uniform(-0.02, 0.02)
                else:
                    disturbance = 0
                
                # PID制御
                steering_correction = pid.update(disturbance)
                
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
                    
                corrections.append(abs(steering_correction))
                speed_diffs.append(abs(left_speed - right_speed))
                
                time.sleep(0.02)  # 50Hz
        
        except KeyboardInterrupt:
            print("テスト中断")
        finally:
            self.et.brake()
            time.sleep(0.5)
        
        # 結果分析
        result = self.analyze_combination_result(kp, kd, limits, base_speed, corrections, speed_diffs, positions)
        return result
    
    def analyze_combination_result(self, kp, kd, limits, base_speed, corrections, speed_diffs, positions):
        """組み合わせ結果の詳細分析"""
        if not corrections:
            return None
        
        # 制御効率指標
        avg_correction = sum(corrections) / len(corrections)
        max_correction = max(corrections)
        correction_variance = sum((c - avg_correction)**2 for c in corrections) / len(corrections)
        
        # スピード効率指標
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
        max_speed_diff = max(speed_diffs)
        
        # 位置安定性
        if len(positions) > 1:
            left_positions = [p[0] for p in positions]
            right_positions = [p[1] for p in positions]
            left_stability = max(left_positions) - min(left_positions)
            right_stability = max(right_positions) - min(right_positions)
            position_stability = (left_stability + right_stability) / 2
        else:
            position_stability = 0
        
        # limits使用効率（重要指標）
        limit_value = limits[1]  # 正の上限値
        limit_usage = max_correction / limit_value if limit_value > 0 else 0
        
        # 総合効率スコア（高速安定性効率 + 制御安定性）
        # より低い値が良い（エネルギー効率重視）
        efficiency_score = (
            avg_correction * 0.4 +           # 制御量効率
            avg_speed_diff * 0.01 * 0.3 +    # スピード効率
            correction_variance * 0.2 +      # 制御一貫性
            limit_usage * 0.1                # limits使用効率
        )
        
        print(f"\n📊 Kp={kp}, Kd={kd}, limits={limits} (速度{base_speed}) 結果:")
        print(f"   平均制御量: {avg_correction:.3f}")
        print(f"   最大制御量: {max_correction:.3f}")
        print(f"   制御分散: {correction_variance:.3f}")
        print(f"   平均速度差: {avg_speed_diff:.2f}")
        print(f"   最大速度差: {max_speed_diff:.2f}")
        print(f"   位置安定性: {position_stability:.1f}")
        print(f"   limits使用率: {limit_usage:.1%}")
        print(f"   効率スコア: {efficiency_score:.3f}")
        
        return {
            'kp': kp,
            'kd': kd,
            'limits': limits,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'correction_variance': correction_variance,
            'avg_speed_diff': avg_speed_diff,
            'max_speed_diff': max_speed_diff,
            'position_stability': position_stability,
            'limit_usage': limit_usage,
            'efficiency_score': efficiency_score
        }
    
    def run_comprehensive_analysis(self):
        """包括的PIDパラメータ分析実行"""
        print("🔬 包括的PIDパラメータ分析テスト")
        print("=" * 70)
        print("⚠️ 安全な場所でテストしてください")
        print("⚠️ Kp、Kd、output_limitsの全組み合わせテスト")
        
        # テスト範囲定義
        kp_values = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # 7種類
        kd_values = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # 7種類
        limits_values = [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-8, 8)]  # 7種類
        speeds = [70, 98]  # HIGH_SPEED_AVOID対応速度
        
        total_combinations = len(kp_values) * len(kd_values) * len(limits_values) * len(speeds)
        print(f"📋 総テスト数: {total_combinations}組み合わせ")
        print(f"   Kp: {len(kp_values)}種類 {kp_values}")
        print(f"   Kd: {len(kd_values)}種類 {kd_values}")
        print(f"   limits: {len(limits_values)}種類 {limits_values}")
        print(f"   速度: {len(speeds)}種類 {speeds}")
        print(f"⏱️ 推定時間: {total_combinations * 5 / 60:.1f}分")
        
        if input("続行しますか？ (y/N): ").lower() != 'y':
            print("テスト中止")
            return
        
        all_results = []
        current_test = 0
        
        for speed in speeds:
            print(f"\n🚀 速度{speed}でのテスト開始")
            for kp in kp_values:
                for kd in kd_values:
                    for limits in limits_values:
                        current_test += 1
                        print(f"\n[{current_test}/{total_combinations}] Kp={kp}, Kd={kd}, limits={limits}")
                        
                        result = self.test_pid_combination(kp, kd, limits, speed, duration=5.0)
                        if result:
                            all_results.append(result)
                        
                        print(f"完了 ({current_test}/{total_combinations})")
        
        # 包括的分析結果
        self.comprehensive_analysis_results(all_results)
    
    def comprehensive_analysis_results(self, results):
        """包括的分析結果の表示"""
        print("\n" + "=" * 100)
        print("🏆 包括的PIDパラメータ分析結果")
        print("=" * 100)
        
        if not results:
            print("❌ 有効な結果がありません")
            return
        
        # 1. 全体ランキング（効率スコア順）
        print("\n🥇 総合効率ランキング（TOP20）:")
        sorted_results = sorted(results, key=lambda x: x['efficiency_score'])
        
        for i, result in enumerate(sorted_results[:20], 1):
            print(f"{i:2d}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}")
            print(f"      速度{result['base_speed']}: 効率スコア{result['efficiency_score']:.3f}")
            print(f"      制御量{result['avg_correction']:.3f}, 速度差{result['avg_speed_diff']:.1f}, limits使用{result['limit_usage']:.1%}")
            print()
        
        # 2. 速度別最適設定
        print("\n📊 速度別最適設定:")
        for speed in [70, 98]:
            speed_results = [r for r in results if r['base_speed'] == speed]
            if speed_results:
                best = min(speed_results, key=lambda x: x['efficiency_score'])
                print(f"  速度{speed}最適: Kp={best['kp']}, Kd={best['kd']}, limits={best['limits']}")
                print(f"              効率スコア{best['efficiency_score']:.3f}, 制御量{best['avg_correction']:.3f}")
        
        # 3. パラメータ別傾向分析
        print("\n📈 パラメータ別傾向分析:")
        
        # Kp別平均効率
        kp_analysis = {}
        for result in results:
            kp = result['kp']
            if kp not in kp_analysis:
                kp_analysis[kp] = []
            kp_analysis[kp].append(result['efficiency_score'])
        
        print("  Kp別平均効率:")
        for kp in sorted(kp_analysis.keys()):
            avg_score = sum(kp_analysis[kp]) / len(kp_analysis[kp])
            print(f"    Kp={kp}: {avg_score:.3f} (サンプル数{len(kp_analysis[kp])})")
        
        # limits別平均効率
        limits_analysis = {}
        for result in results:
            limits = result['limits']
            if limits not in limits_analysis:
                limits_analysis[limits] = []
            limits_analysis[limits].append(result['efficiency_score'])
        
        print("  limits別平均効率:")
        for limits in sorted(limits_analysis.keys(), key=lambda x: x[1]):
            avg_score = sum(limits_analysis[limits]) / len(limits_analysis[limits])
            print(f"    limits={limits}: {avg_score:.3f} (サンプル数{len(limits_analysis[limits])})")
        
        # 4. 最終推奨設定
        print("\n🎯 最終推奨設定:")
        overall_best = sorted_results[0]
        print(f"  🥇 総合最優秀: Kp={overall_best['kp']}, Kd={overall_best['kd']}, limits={overall_best['limits']}")
        print(f"      速度{overall_best['base_speed']}: 効率スコア{overall_best['efficiency_score']:.3f}")
        print(f"      制御量{overall_best['avg_correction']:.3f}, 速度差{overall_best['avg_speed_diff']:.1f}")
        print(f"      limits使用率{overall_best['limit_usage']:.1%}")
        
        # 両速度対応の最適設定を探す
        print("\n🚀 両速度対応分析:")
        dual_speed_analysis = {}
        for result in results:
            key = (result['kp'], result['kd'], result['limits'])
            if key not in dual_speed_analysis:
                dual_speed_analysis[key] = []
            dual_speed_analysis[key].append(result)
        
        # 両速度でテストされた設定のみを評価
        dual_speed_candidates = []
        for key, speed_results in dual_speed_analysis.items():
            if len(speed_results) == 2:  # 70と98両方でテスト済み
                avg_efficiency = sum(r['efficiency_score'] for r in speed_results) / 2
                consistency = 1.0 - abs(speed_results[0]['efficiency_score'] - speed_results[1]['efficiency_score']) / max(r['efficiency_score'] for r in speed_results)
                dual_score = avg_efficiency * (0.7 + 0.3 * consistency)  # 一貫性重視
                
                dual_speed_candidates.append({
                    'kp': key[0],
                    'kd': key[1],
                    'limits': key[2],
                    'dual_score': dual_score,
                    'avg_efficiency': avg_efficiency,
                    'consistency': consistency,
                    'results': speed_results
                })
        
        if dual_speed_candidates:
            dual_speed_candidates.sort(key=lambda x: x['dual_score'])
            best_dual = dual_speed_candidates[0]
            
            print(f"  🎯 両速度最適: Kp={best_dual['kp']}, Kd={best_dual['kd']}, limits={best_dual['limits']}")
            print(f"      両速度スコア: {best_dual['dual_score']:.3f}")
            print(f"      平均効率: {best_dual['avg_efficiency']:.3f}")
            print(f"      一貫性: {best_dual['consistency']:.3f}")
            
            for result in best_dual['results']:
                print(f"        速度{result['base_speed']}: 効率{result['efficiency_score']:.3f}, 制御量{result['avg_correction']:.3f}")
        
        print(f"\n📋 結論:")
        print(f"   HIGH_SPEED_AVOIDモードの推奨設定:")
        if dual_speed_candidates:
            best_dual = dual_speed_candidates[0]
            print(f"   Kp={best_dual['kp']}, Kd={best_dual['kd']}, limits={best_dual['limits']}")
            print(f"   (両速度70/98で最適バランス、効率重視)")
        else:
            print(f"   Kp={overall_best['kp']}, Kd={overall_best['kd']}, limits={overall_best['limits']}")
            print(f"   (単一速度での最適設定)")

def main():
    analyzer = ComprehensivePIDLimitsAnalysis()
    try:
        analyzer.run_comprehensive_analysis()
    finally:
        analyzer.et.stop()

if __name__ == "__main__":
    main()
