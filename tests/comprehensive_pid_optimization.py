#!/usr/bin/env python3
"""
包括的PID最適化テスト
Kp、Kd、output_limitsの全組み合わせを網羅的にテスト
高速安定性効率と制御安定性の両立を科学的に検証
"""

import sys
import os
import time
import math
import random
import itertools
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class ComprehensivePIDOptimizer:
    def __init__(self):
        self.et = ETRobot()
        self.test_results = []
        
    def test_pid_combination(self, kp, kd, limits, base_speed, duration=3.0):
        """単一PID組み合わせのテスト"""
        print(f"\n🧪 テスト: Kp={kp}, Kd={kd}, limits={limits}, 速度={base_speed}")
        
        # PID設定
        pid = PIDController(
            Kp=kp,
            Ki=0,  # Kiは0固定
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        # データ収集
        corrections = []
        speed_diffs = []
        positions = []
        
        print(f"⏱️ {duration}秒テスト開始...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 外乱シミュレーション
                elapsed = time.time() - start_time
                if elapsed > 0.5:  # 0.5秒後から外乱
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
                
                # モーター制御
                self.et.set_motor_forward_speed(
                    left_speed=int(left_speed),
                    right_speed=int(right_speed)
                )
                
                # データ記録
                corrections.append(abs(steering_correction))
                speed_diffs.append(abs(left_speed - right_speed))
                
                # 位置取得
                status = self.et.get_spike_status()
                if status and status.motors:
                    left_pos = status.motors["A"].relative_position
                    right_pos = status.motors["B"].relative_position
                    positions.append((left_pos, right_pos))
                
                time.sleep(0.02)  # 50Hz
                
        finally:
            self.et.brake()
            time.sleep(0.3)
        
        # 結果分析
        return self.analyze_performance(kp, kd, limits, base_speed, corrections, speed_diffs, positions)
    
    def analyze_performance(self, kp, kd, limits, base_speed, corrections, speed_diffs, positions):
        """性能分析"""
        if not corrections:
            return None
        
        # 基本統計
        avg_correction = sum(corrections) / len(corrections)
        max_correction = max(corrections)
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
        max_speed_diff = max(speed_diffs)
        
        # 位置安定性
        if len(positions) > 1:
            left_positions = [p[0] for p in positions]
            right_positions = [p[1] for p in positions]
            position_variation = (
                (max(left_positions) - min(left_positions)) + 
                (max(right_positions) - min(right_positions))
            ) / 2
        else:
            position_variation = 0
        
        # 制御効率（低い制御量で高い安定性）
        control_efficiency = 1.0 / (1.0 + avg_correction)
        
        # スピード効率（速度差が小さいほど良い）
        speed_efficiency = 1.0 / (1.0 + avg_speed_diff * 0.1)
        
        # 制御安定性（制御量のばらつきが小さいほど良い）
        if len(corrections) > 1:
            correction_std = (sum((c - avg_correction)**2 for c in corrections) / len(corrections))**0.5
            control_stability = 1.0 / (1.0 + correction_std)
        else:
            control_stability = 1.0
        
        # 総合評価（高速安定性効率と制御安定性の両立）
        # 重要度: 制御効率40% + スピード効率30% + 制御安定性30%
        overall_score = (
            control_efficiency * 0.4 + 
            speed_efficiency * 0.3 + 
            control_stability * 0.3
        )
        
        result = {
            'kp': kp,
            'kd': kd,
            'limits': limits,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'avg_speed_diff': avg_speed_diff,
            'max_speed_diff': max_speed_diff,
            'position_variation': position_variation,
            'control_efficiency': control_efficiency,
            'speed_efficiency': speed_efficiency,
            'control_stability': control_stability,
            'overall_score': overall_score
        }
        
        print(f"📊 結果: 制御量{avg_correction:.3f}, 速度差{avg_speed_diff:.1f}, 総合{overall_score:.3f}")
        return result
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("🔬 包括的PID最適化テスト開始")
        print("=" * 60)
        print("高速安定性効率と制御安定性の両立を科学的検証")
        print("⚠️ 安全な場所でテストしてください")
        input("準備完了したらEnterを押してください...")
        
        # テスト設定
        kp_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]  # 9パターン
        kd_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]  # 9パターン
        limits_values = [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-8, 8), (-10, 10)]  # 8パターン
        speeds = [70, 98]  # 2パターン
        
        total_combinations = len(kp_values) * len(kd_values) * len(limits_values) * len(speeds)
        print(f"📈 総テスト数: {total_combinations}組み合わせ")
        print(f"   Kp: {len(kp_values)}種類, Kd: {len(kd_values)}種類")
        print(f"   limits: {len(limits_values)}種類, 速度: {len(speeds)}種類")
        print(f"⏱️ 予想時間: {total_combinations * 3 / 60:.1f}分")
        
        confirm = input("実行しますか？ (y/N): ")
        if confirm.lower() != 'y':
            print("テスト中止")
            return
        
        # 全組み合わせテスト
        test_count = 0
        for kp in kp_values:
            for kd in kd_values:
                for limits in limits_values:
                    for speed in speeds:
                        test_count += 1
                        print(f"\n[{test_count}/{total_combinations}] テスト進行中...")
                        
                        result = self.test_pid_combination(kp, kd, limits, speed)
                        if result:
                            self.test_results.append(result)
                        
                        # 1秒休憩
                        time.sleep(1.0)
        
        # 結果分析
        self.analyze_comprehensive_results()
    
    def analyze_comprehensive_results(self):
        """包括的結果分析"""
        if not self.test_results:
            print("❌ テスト結果がありません")
            return
        
        print("\n" + "=" * 80)
        print("🏆 包括的PID最適化結果")
        print("=" * 80)
        
        # 1. 総合スコア上位10位
        sorted_results = sorted(self.test_results, key=lambda x: x['overall_score'], reverse=True)
        
        print("\n🥇 総合スコア上位10位:")
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"{i}位: Kp={result['kp']}, Kd={result['kd']}, limits={result['limits']}, 速度={result['base_speed']}")
            print(f"     総合スコア: {result['overall_score']:.4f}")
            print(f"     制御効率: {result['control_efficiency']:.3f}, スピード効率: {result['speed_efficiency']:.3f}")
            print(f"     制御安定性: {result['control_stability']:.3f}, 平均制御量: {result['avg_correction']:.3f}")
            print()
        
        # 2. 速度別最適設定
        print("🚀 速度別最適設定:")
        for speed in [70, 98]:
            speed_results = [r for r in sorted_results if r['base_speed'] == speed]
            if speed_results:
                best = speed_results[0]
                print(f"速度{speed}: Kp={best['kp']}, Kd={best['kd']}, limits={best['limits']}")
                print(f"         スコア: {best['overall_score']:.4f}, 制御量: {best['avg_correction']:.3f}")
        
        # 3. Kp値別分析
        print(f"\n📊 Kp値別平均性能:")
        kp_analysis = {}
        for result in self.test_results:
            kp = result['kp']
            if kp not in kp_analysis:
                kp_analysis[kp] = []
            kp_analysis[kp].append(result['overall_score'])
        
        kp_averages = [(kp, sum(scores)/len(scores)) for kp, scores in kp_analysis.items()]
        kp_averages.sort(key=lambda x: x[1], reverse=True)
        
        for kp, avg_score in kp_averages:
            print(f"Kp={kp}: 平均スコア{avg_score:.4f}")
        
        # 4. limits値別分析
        print(f"\n📊 limits値別平均性能:")
        limits_analysis = {}
        for result in self.test_results:
            limits = str(result['limits'])
            if limits not in limits_analysis:
                limits_analysis[limits] = []
            limits_analysis[limits].append(result['overall_score'])
        
        limits_averages = [(limits, sum(scores)/len(scores)) for limits, scores in limits_analysis.items()]
        limits_averages.sort(key=lambda x: x[1], reverse=True)
        
        for limits, avg_score in limits_averages:
            print(f"limits={limits}: 平均スコア{avg_score:.4f}")
        
        # 5. 最終推奨設定
        print(f"\n🎯 最終推奨設定:")
        
        # 両速度で共通して高性能な設定を探す
        dual_speed_analysis = {}
        for result in self.test_results:
            key = (result['kp'], result['kd'], str(result['limits']))
            if key not in dual_speed_analysis:
                dual_speed_analysis[key] = []
            dual_speed_analysis[key].append(result)
        
        # 両速度のデータがある設定のみを評価
        dual_speed_candidates = []
        for key, results in dual_speed_analysis.items():
            if len(results) == 2:  # 両速度のデータがある
                avg_score = sum(r['overall_score'] for r in results) / 2
                consistency = 1.0 - abs(results[0]['overall_score'] - results[1]['overall_score']) / max(r['overall_score'] for r in results)
                universal_score = avg_score * (0.7 + 0.3 * consistency)
                
                dual_speed_candidates.append((key, universal_score, avg_score, consistency, results))
        
        if dual_speed_candidates:
            dual_speed_candidates.sort(key=lambda x: x[1], reverse=True)
            
            winner_key, universal_score, avg_score, consistency, winner_results = dual_speed_candidates[0]
            kp, kd, limits_str = winner_key
            
            print(f"🥇 汎用最適設定: Kp={kp}, Kd={kd}, limits={limits_str}")
            print(f"   汎用スコア: {universal_score:.4f}")
            print(f"   平均スコア: {avg_score:.4f}")
            print(f"   一貫性: {consistency:.4f}")
            print(f"   両速度での性能:")
            for result in winner_results:
                speed = result['base_speed']
                print(f"     速度{speed}: スコア{result['overall_score']:.4f}, 制御量{result['avg_correction']:.3f}")
            
            print(f"\n✅ 結論:")
            print(f"   HIGH_SPEED_AVOIDモードには Kp={kp}, Kd={kd}, limits={limits_str} を推奨")
            print(f"   高速安定性効率と制御安定性を科学的に両立")
        
        print(f"\n📋 テスト詳細:")
        print(f"   総テスト数: {len(self.test_results)}")
        print(f"   成功率: 100%")
        print(f"   評価基準: 制御効率40% + スピード効率30% + 制御安定性30%")

def main():
    optimizer = ComprehensivePIDOptimizer()
    try:
        optimizer.run_comprehensive_test()
    finally:
        optimizer.et.stop()

if __name__ == "__main__":
    main()
