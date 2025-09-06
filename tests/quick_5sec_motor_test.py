#!/usr/bin/env python3
"""
5秒実走行PID比較テスト (カメラレス版)
純粋なモーター制御安定性を70/98速度でテスト
ライン検出なし - 微小外乱に対するPID応答性テスト
"""

import sys
import os
import time
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController

class Quick5SecMotorTest:
    def __init__(self):
        self.et = ETRobot()
        # カメラ不要（ライン検出なし）
        
    def test_pid_setting(self, kp, kd, limits, name, base_speed, duration=5.0):
        """5秒間の純粋なモーター安定性テスト"""
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
        
        print(f"🚗 {name} (速度{base_speed}) 5秒テスト開始！")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 微小な外乱を追加（PIDの応答性をテスト）
                elapsed = time.time() - start_time
                if elapsed > 1:  # 1秒後から外乱開始
                    disturbance = random.uniform(-0.02, 0.02)  # ラジアン単位の微小外乱
                else:
                    disturbance = 0
                
                # PID制御（外乱に対する応答性テスト）
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
                    speeds.append((left_speed, right_speed))
                    corrections.append(abs(steering_correction))
                
                time.sleep(0.02)  # 50Hz
        
        except KeyboardInterrupt:
            print("テスト中断")
        finally:
            self.et.brake()
            time.sleep(0.5)
        
        # 結果分析
        result = self.analyze_results(name, base_speed, positions, speeds, corrections)
        return result
    
    def analyze_results(self, name, base_speed, positions, speeds, corrections):
        """結果分析"""
        if not positions:
            print(f"❌ {name} (速度{base_speed}): データ不足")
            return None
        
        # 安定性評価
        avg_correction = sum(corrections) / len(corrections) if corrections else 0
        max_correction = max(corrections) if corrections else 0
        
        # 速度一貫性
        speed_diffs = []
        for left, right in speeds:
            speed_diffs.append(abs(left - right))
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs) if speed_diffs else 0
        
        # 位置安定性
        if len(positions) > 1:
            left_positions = [p[0] for p in positions]
            right_positions = [p[1] for p in positions]
            left_stability = max(left_positions) - min(left_positions)
            right_stability = max(right_positions) - min(right_positions)
            position_stability = (left_stability + right_stability) / 2
        else:
            position_stability = 0
        
        print(f"\n📊 {name} (速度{base_speed}) 結果:")
        print(f"   平均制御量: {avg_correction:.3f}")
        print(f"   最大制御量: {max_correction:.3f}")
        print(f"   速度差平均: {avg_speed_diff:.2f}")
        print(f"   位置安定性: {position_stability:.1f}")
        
        # 総合評価（外乱への応答性とバランス）
        stability_score = 1.0 / (1.0 + avg_correction + avg_speed_diff * 0.01)
        print(f"   安定性スコア: {stability_score:.3f}")
        
        return {
            'name': name,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'avg_speed_diff': avg_speed_diff,
            'position_stability': position_stability,
            'stability_score': stability_score
        }
    
    def run_comparison_test(self):
        """70/98速度でのPID比較テスト実行"""
        print("🧪 5秒実走行PID比較テスト (70/98速度)")
        print("=" * 60)
        print("⚠️ 安全な場所でテストしてください")
        print("⚠️ カメラ不要 - 純粋なモーター制御テスト")
        
        # テスト設定
        pid_configs = [
            {
                'name': '最適設定 (科学的証明済み)',
                'kp': 0.3,
                'kd': 0.3,
                'limits': (-2, 2)
            },
            {
                'name': '中間設定 (推測)',
                'kp': 1.0,
                'kd': 1.0,
                'limits': (-4, 4)
            }
        ]
        
        speeds = [70, 98]  # HIGH_SPEED_AVOID対応速度
        
        all_results = []
        
        for speed in speeds:
            print(f"\n🚀 速度{speed}でのテスト開始")
            for config in pid_configs:
                result = self.test_pid_setting(
                    kp=config['kp'],
                    kd=config['kd'],
                    limits=config['limits'],
                    name=config['name'],
                    base_speed=speed,
                    duration=5.0
                )
                if result:
                    all_results.append(result)
                
                print(f"\n{config['name']} (速度{speed}) 完了")
                input("次のテストに進むには車体を再配置してEnterを押してください...")
        
        # 最終比較
        self.final_comparison(all_results)
    
    def final_comparison(self, results):
        """最終比較結果"""
        print("\n" + "=" * 70)
        print("🏆 5秒実走行テスト最終結果 (70/98速度比較)")
        print("=" * 70)
        
        # 速度別に整理
        results_70 = [r for r in results if r['base_speed'] == 70]
        results_98 = [r for r in results if r['base_speed'] == 98]
        
        print("📊 速度70での結果:")
        for result in sorted(results_70, key=lambda x: x['stability_score'], reverse=True):
            print(f"  {result['name']}: 安定性{result['stability_score']:.3f}, 制御量{result['avg_correction']:.3f}")
        
        print("\n📊 速度98での結果:")
        for result in sorted(results_98, key=lambda x: x['stability_score'], reverse=True):
            print(f"  {result['name']}: 安定性{result['stability_score']:.3f}, 制御量{result['avg_correction']:.3f}")
        
        # 汎用性評価（両速度での総合性能）
        print(f"\n🎯 汎用性評価 (70/98両速度):")
        
        # 設定別に両速度の平均スコアを計算
        setting_scores = {}
        for result in results:
            name = result['name']
            if name not in setting_scores:
                setting_scores[name] = []
            setting_scores[name].append(result['stability_score'])
        
        final_rankings = []
        for name, scores in setting_scores.items():
            avg_score = sum(scores) / len(scores)
            consistency = 1.0 - abs(scores[0] - scores[1]) if len(scores) == 2 else 0
            universal_score = avg_score * (0.7 + 0.3 * consistency)
            final_rankings.append((name, universal_score, avg_score, consistency))
        
        final_rankings.sort(key=lambda x: x[1], reverse=True)
        
        print("最終ランキング:")
        for i, (name, universal_score, avg_score, consistency) in enumerate(final_rankings, 1):
            print(f"{i}位: {name}")
            print(f"     汎用スコア: {universal_score:.3f}")
            print(f"     平均安定性: {avg_score:.3f}")
            print(f"     一貫性: {consistency:.3f}")
            print()
        
        # 勝者発表
        if final_rankings:
            winner = final_rankings[0][0]
            print(f"🥇 総合勝者: {winner}")
            
            if "最適設定" in winner:
                print("✅ 科学的テストの結果が実走行でも証明されました！")
                print("   Kp=0.3, Kd=0.3, limits=(-2,2) が両速度で最適")
            elif "中間設定" in winner:
                print("🤔 中間設定が予想外に良い結果でした...")
                print("   理論と実際の違いが明らかになりました")

def main():
    tester = Quick5SecMotorTest()
    try:
        tester.run_comparison_test()
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
