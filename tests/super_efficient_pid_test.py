#!/usr/bin/env python3
"""
超効率化PIDテスト
Kd=0.5固定で、Kpとlimitsのみをピンポイントでテスト
車輪浮かせ、完全自動、2分で完了
"""

import sys
import os
import time
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class SuperEfficientPIDTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_airborne(self, kp, kd, limits, base_speed, duration=1.0):
        """超高速PIDテスト（1秒）"""
        pid = PIDController(
            Kp=kp,
            Ki=0,
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        corrections = []
        speed_diffs = []
        
        print(f"⚡ Kp={kp}, Kd={kd}, limits={limits}, 速度={base_speed} ", end="")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                if elapsed > 0.3:  # 早期外乱開始
                    disturbance = random.uniform(-0.03, 0.03)
                else:
                    disturbance = 0
                
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
                
                time.sleep(0.02)
                
        finally:
            self.et.brake()
            time.sleep(0.1)
        
        # 簡略分析
        if not corrections:
            return None
        
        avg_correction = sum(corrections) / len(corrections)
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
        
        # 効率スコア（制御量とスピード効率）
        efficiency_score = 1.0 / (1.0 + avg_correction + avg_speed_diff * 0.1)
        
        print(f"→ 制御{avg_correction:.2f}, 効率{efficiency_score:.3f}")
        
        return {
            'kp': kp,
            'kd': kd,
            'limits': limits,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'avg_speed_diff': avg_speed_diff,
            'efficiency_score': efficiency_score
        }
    
    def run_super_efficient_test(self):
        """超効率化テスト実行"""
        print("⚡ 超効率化PIDテスト")
        print("=" * 40)
        print("🔍 Kd=0含む全パターン検証")
        print("🎯 1分で結論を出す")
        
        # 超効率化範囲（Kd=0を含む検証）
        kd_values = [0, 0.3, 0.5, 1.0]  # Kd=0も検証、0.3は従来値
        kp_values = [0.5, 1.0, 1.5]  # 3種類に絞り込み
        limits_values = [(-3, 3), (-4, 4), (-5, 5)]  # 3種類
        speeds = [70, 98]  # 2種類
        
        total_tests = len(kd_values) * len(kp_values) * len(limits_values) * len(speeds)
        print(f"📊 結論重視テスト数: {total_tests}")
        print(f"   Kd: {kd_values} (Kd=0検証含む)")
        print(f"   Kp: {kp_values}")
        print(f"   limits: {limits_values}")
        print(f"⏱️ 推定時間: {total_tests * 1.0 / 60:.1f}分")
        
        if input("車輪浮かせて結論重視テスト開始？ (y/N): ").lower() != 'y':
            return
        
        print(f"\n🚀 Kd=0含む全パターン、超高速テスト開始！")
        all_results = []
        test_count = 0
        
        for speed in speeds:
            print(f"\n📈 速度{speed}:")
            for kd in kd_values:
                for kp in kp_values:
                    for limits in limits_values:
                        test_count += 1
                        print(f"[{test_count:2d}/{total_tests}] ", end="")
                        
                        result = self.test_pid_airborne(kp=kp, kd=kd, limits=limits, base_speed=speed)
                        if result:
                            all_results.append(result)
                        
                        time.sleep(0.1)  # 最小休憩
        
        # 結果分析
        self.analyze_super_efficient_results(all_results)
    
    def analyze_super_efficient_results(self, results):
        """超効率化結果分析"""
        print("\n" + "=" * 60)
        print("🏆 超効率化テスト結果")
        print("=" * 60)
        
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
    tester = SuperEfficientPIDTest()
    try:
        tester.run_super_efficient_test()
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
