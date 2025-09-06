#!/usr/bin/env python3
"""
車輪浮かせPID自動テスト
机上でモーターを浮かせた状態で全組み合わせを自動テスト
エンター入力なし、完全自動化
"""

import sys
import os
import time
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class AirborneAutoPIDTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_airborne(self, kp, kd, limits, base_speed, duration=3.0):
        """車輪浮かせでの自動PIDテスト（エンター入力なし）"""
        # PID設定
        pid = PIDController(
            Kp=kp,
            Ki=0,
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        # データ収集用
        corrections = []
        speed_diffs = []
        motor_loads = []
        
        print(f"🔬 自動テスト: Kp={kp}, Kd={kd}, limits={limits}, 速度={base_speed}")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 外乱シミュレーション
                elapsed = time.time() - start_time
                if elapsed > 0.5:
                    disturbance = random.uniform(-0.03, 0.03)
                else:
                    disturbance = 0
                
                # PID制御
                steering_correction = pid.update(disturbance)
                
                # 速度計算
                left_speed = base_speed - steering_correction
                right_speed = base_speed + steering_correction
                
                # 速度制限
                left_speed = max(0, min(255, left_speed))
                right_speed = max(0, min(255, right_speed))
                
                # モーター制御（車輪浮かせ状態）
                self.et.set_motor_forward_speed(
                    left_speed=int(left_speed),
                    right_speed=int(right_speed)
                )
                
                # データ記録
                corrections.append(abs(steering_correction))
                speed_diffs.append(abs(left_speed - right_speed))
                
                # モーター負荷取得（浮かせ状態でも取得可能）
                status = self.et.get_spike_status()
                if status and status.motors:
                    left_power = getattr(status.motors["A"], "power", 0)
                    right_power = getattr(status.motors["B"], "power", 0)
                    motor_loads.append((abs(left_power), abs(right_power)))
                
                time.sleep(0.02)  # 50Hz
                
        finally:
            self.et.brake()
            time.sleep(0.1)
        
        # 結果分析
        return self.analyze_airborne_result(kp, kd, limits, base_speed, corrections, speed_diffs, motor_loads)
    
    def analyze_airborne_result(self, kp, kd, limits, base_speed, corrections, speed_diffs, motor_loads):
        """車輪浮かせテスト結果の分析"""
        if not corrections:
            return None
        
        # 基本統計
        avg_correction = sum(corrections) / len(corrections)
        max_correction = max(corrections)
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
        
        # モーター負荷効率
        if motor_loads:
            avg_left_load = sum(load[0] for load in motor_loads) / len(motor_loads)
            avg_right_load = sum(load[1] for load in motor_loads) / len(motor_loads)
            avg_motor_load = (avg_left_load + avg_right_load) / 2
        else:
            avg_motor_load = 0
        
        # limits使用効率
        limit_value = limits[1]
        limit_usage = max_correction / limit_value if limit_value > 0 else 0
        
        # 制御効率（低制御量、低負荷が良い）
        control_efficiency = 1.0 / (1.0 + avg_correction + avg_motor_load * 0.01)
        
        # スピード効率（速度差小）
        speed_efficiency = 1.0 / (1.0 + avg_speed_diff * 0.1)
        
        # 総合効率スコア
        overall_score = control_efficiency * 0.6 + speed_efficiency * 0.4
        
        result = {
            'kp': kp,
            'kd': kd,
            'limits': limits,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'avg_speed_diff': avg_speed_diff,
            'avg_motor_load': avg_motor_load,
            'limit_usage': limit_usage,
            'control_efficiency': control_efficiency,
            'speed_efficiency': speed_efficiency,
            'overall_score': overall_score
        }
        
        print(f"   結果: 制御{avg_correction:.2f}, 速度差{avg_speed_diff:.1f}, 負荷{avg_motor_load:.1f}, スコア{overall_score:.3f}")
        return result
    
    def run_airborne_auto_test(self):
        """車輪浮かせ全自動テスト実行"""
        print("🚁 車輪浮かせ全自動PIDテスト")
        print("=" * 50)
        print("⚠️ 車輪を浮かせた状態でテストします")
        print("⚠️ 完全自動化、エンター入力不要")
        
        # 戦略的テスト範囲（実測結果を基に絞り込み）
        # 実測で中間設定Kp=1.0が優秀だったので、その周辺を重点的に
        kp_values = [0.5, 1.0, 1.5, 2.0]  # 4種類（1.0周辺重点）
        kd_values = [0.5, 1.0, 1.5, 2.0]  # 4種類（1.0周辺重点）
        # 実測で(-4,4)が良好だったので、その周辺を重点的に
        limits_values = [(-3, 3), (-4, 4), (-5, 5), (-6, 6)]  # 4種類
        speeds = [70, 98]  # 2種類
        
        total_tests = len(kp_values) * len(kd_values) * len(limits_values) * len(speeds)
        print(f"📊 戦略的テスト数: {total_tests} (効率化済み)")
        print(f"   Kp: {kp_values} (1.0周辺重点)")
        print(f"   Kd: {kd_values} (1.0周辺重点)")  
        print(f"   limits: {limits_values} ((-4,4)周辺重点)")
        print(f"⏱️ 推定時間: {total_tests * 2 / 60:.1f}分")
        
        if input("車輪を浮かせて開始しますか？ (y/N): ").lower() != 'y':
            print("テスト中止")
            return
        
        print("\n🚀 全自動テスト開始！")
        all_results = []
        test_count = 0
        
        for speed in speeds:
            for kp in kp_values:
                for kd in kd_values:
                    for limits in limits_values:
                        test_count += 1
                        print(f"[{test_count:3d}/{total_tests}] ", end="")
                        
                        result = self.test_pid_airborne(kp, kd, limits, speed, duration=2.0)  # 2秒に短縮
                        if result:
                            all_results.append(result)
                        
                        # 短い休憩
                        time.sleep(0.5)
        
        # 結果分析
        self.analyze_comprehensive_airborne_results(all_results)
    
    def analyze_comprehensive_airborne_results(self, results):
        """車輪浮かせテスト包括結果分析"""
        print("\n" + "=" * 80)
        print("🏆 車輪浮かせ全自動テスト結果")
        print("=" * 80)
        
        if not results:
            print("❌ 有効な結果がありません")
            return
        
        # 1. 総合ランキング
        sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)
        
        print("\n🥇 総合効率ランキング（TOP15）:")
        for i, result in enumerate(sorted_results[:15], 1):
            print(f"{i:2d}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}")
            print(f"      速度{result['base_speed']}: スコア{result['overall_score']:.3f}")
            print(f"      制御量{result['avg_correction']:.2f}, 速度差{result['avg_speed_diff']:.1f}, 負荷{result['avg_motor_load']:.1f}")
        
        # 2. 両速度対応分析
        print(f"\n🚀 両速度対応分析:")
        dual_speed_analysis = {}
        for result in results:
            key = (result['kp'], result['kd'], result['limits'])
            if key not in dual_speed_analysis:
                dual_speed_analysis[key] = []
            dual_speed_analysis[key].append(result)
        
        dual_candidates = []
        for key, speed_results in dual_speed_analysis.items():
            if len(speed_results) == 2:  # 両速度
                avg_score = sum(r['overall_score'] for r in speed_results) / 2
                consistency = 1.0 - abs(speed_results[0]['overall_score'] - speed_results[1]['overall_score']) / max(r['overall_score'] for r in speed_results)
                dual_score = avg_score * (0.8 + 0.2 * consistency)
                
                dual_candidates.append({
                    'kp': key[0],
                    'kd': key[1], 
                    'limits': key[2],
                    'dual_score': dual_score,
                    'avg_score': avg_score,
                    'consistency': consistency,
                    'results': speed_results
                })
        
        dual_candidates.sort(key=lambda x: x['dual_score'], reverse=True)
        
        print("🎯 両速度総合ランキング（TOP10）:")
        for i, candidate in enumerate(dual_candidates[:10], 1):
            print(f"{i:2d}位: Kp={candidate['kp']}, Kd={candidate['kd']}, limits={candidate['limits']}")
            print(f"      両速度スコア: {candidate['dual_score']:.3f}")
            print(f"      平均: {candidate['avg_score']:.3f}, 一貫性: {candidate['consistency']:.3f}")
            for result in candidate['results']:
                print(f"        速度{result['base_speed']}: {result['overall_score']:.3f}")
        
        # 3. パラメータ別傾向
        print(f"\n📈 パラメータ別最適傾向:")
        
        # Kp別平均
        kp_scores = {}
        for result in results:
            if result['kp'] not in kp_scores:
                kp_scores[result['kp']] = []
            kp_scores[result['kp']].append(result['overall_score'])
        
        print("  Kp値別平均スコア:")
        for kp in sorted(kp_scores.keys()):
            avg = sum(kp_scores[kp]) / len(kp_scores[kp])
            print(f"    Kp={kp}: {avg:.3f}")
        
        # limits別平均
        limits_scores = {}
        for result in results:
            limits_key = result['limits']
            if limits_key not in limits_scores:
                limits_scores[limits_key] = []
            limits_scores[limits_key].append(result['overall_score'])
        
        print("  limits別平均スコア:")
        for limits in sorted(limits_scores.keys(), key=lambda x: x[1]):
            avg = sum(limits_scores[limits]) / len(limits_scores[limits])
            print(f"    limits={limits}: {avg:.3f}")
        
        # 4. 最終推奨
        if dual_candidates:
            best = dual_candidates[0]
            print(f"\n🏆 最終推奨設定:")
            print(f"   Kp={best['kp']}, Kd={best['kd']}, limits={best['limits']}")
            print(f"   両速度スコア: {best['dual_score']:.3f}")
            print(f"   車輪浮かせテストでの最適解")
            
            print(f"\n📋 詳細性能:")
            for result in best['results']:
                print(f"   速度{result['base_speed']}: スコア{result['overall_score']:.3f}")
                print(f"     制御量{result['avg_correction']:.2f}, 速度差{result['avg_speed_diff']:.1f}")
                print(f"     モーター負荷{result['avg_motor_load']:.1f}, limits使用{result['limit_usage']:.1%}")
            
            print(f"\n✅ 結論:")
            print(f"   HIGH_SPEED_AVOIDモード推奨設定:")
            print(f"   Kp={best['kp']}, Kd={best['kd']}, output_limits={best['limits']}")
            print(f"   (車輪浮かせ全自動テストによる科学的最適解)")

def main():
    tester = AirborneAutoPIDTest()
    try:
        tester.run_airborne_auto_test()
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
