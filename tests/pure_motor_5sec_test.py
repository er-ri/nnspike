#!/usr/bin/env python3
"""
5秒Pure Motor安定性テスト
カメラなし・ライン検出なしで純粋なモーター制御性能を評価
HIGH_SPEED_BASE 70と98の両方でテスト
"""

import sys
import os
import time
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

class Pure5SecMotorTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_motor_stability(self, base_speed, name, duration=5.0):
        """純粋なモーター安定性テスト（外乱あり）"""
        print(f"\n=== {name} (速度{base_speed}) 安定性テスト ===")
        print("車体を平らな場所にセットして準備...")
        input("準備完了したらEnterを押してください...")
        
        # データ収集用
        positions = []
        disturbances = []
        
        print(f"🚗 {name} 5秒テスト開始！（意図的外乱あり）")
        start_time = time.time()
        
        try:
            step = 0
            while time.time() - start_time < duration:
                # 意図的な外乱を与える（実際のライン追従をシミュレート）
                if step % 50 == 0:  # 1秒ごと
                    # -3から+3の範囲でランダムな外乱
                    disturbance = random.uniform(-3, 3)
                    disturbances.append(disturbance)
                else:
                    disturbance = 0
                
                # 外乱を加えた速度設定
                left_speed = base_speed + disturbance
                right_speed = base_speed - disturbance
                
                # 速度制限
                left_speed = max(10, min(120, left_speed))
                right_speed = max(10, min(120, right_speed))
                
                # モーター制御
                self.et.set_motor_forward_speed(
                    left_speed=int(left_speed),
                    right_speed=int(right_speed)
                )
                
                # 位置データ記録
                status = self.et.get_spike_status()
                if status and status.motors:
                    left_pos = status.motors["A"].relative_position
                    right_pos = status.motors["B"].relative_position
                    positions.append((left_pos, right_pos))
                
                step += 1
                time.sleep(0.02)  # 50Hz
        
        except KeyboardInterrupt:
            print("テスト中断")
        finally:
            self.et.brake()
            time.sleep(0.5)
        
        # 結果分析
        result = self.analyze_motor_results(name, base_speed, positions, disturbances)
        return result
    
    def analyze_motor_results(self, name, base_speed, positions, disturbances):
        """純粋なモーター結果分析"""
        if not positions:
            print(f"❌ {name}: データ不足")
            return None
        
        # 位置安定性分析
        left_positions = [p[0] for p in positions]
        right_positions = [p[1] for p in positions]
        
        # 位置変化の標準偏差（安定性指標）
        import statistics
        left_std = statistics.stdev(left_positions) if len(left_positions) > 1 else 0
        right_std = statistics.stdev(right_positions) if len(right_positions) > 1 else 0
        position_stability = (left_std + right_std) / 2
        
        # 左右差分析
        position_diffs = [abs(l - r) for l, r in positions]
        avg_diff = sum(position_diffs) / len(position_diffs) if position_diffs else 0
        max_diff = max(position_diffs) if position_diffs else 0
        
        # 直進性評価（理想的には左右同期）
        left_total = left_positions[-1] - left_positions[0] if len(left_positions) > 1 else 0
        right_total = right_positions[-1] - right_positions[0] if len(right_positions) > 1 else 0
        straightness = abs(left_total - right_total)
        
        print(f"\n📊 {name} (速度{base_speed}) 結果:")
        print(f"   位置安定性: {position_stability:.1f}")
        print(f"   左右差平均: {avg_diff:.1f}")
        print(f"   左右差最大: {max_diff:.1f}")
        print(f"   直進性: {straightness:.1f} (小さいほど良い)")
        
        # 総合スコア（小さいほど良い）
        stability_score = position_stability + avg_diff * 0.5 + straightness * 0.1
        print(f"   総合安定性: {stability_score:.2f} (小さいほど良い)")
        
        return {
            'speed': base_speed,
            'position_stability': position_stability,
            'avg_diff': avg_diff,
            'max_diff': max_diff,
            'straightness': straightness,
            'stability_score': stability_score
        }
    
    def run_dual_speed_test(self):
        """70と98の両速度でテスト"""
        print("🧪 Pure Motor 5秒安定性テスト")
        print("=" * 50)
        print("⚠️ カメラ・ライン検出なし、純粋なモーター制御テスト")
        print("⚠️ 平らな場所で実行してください")
        
        # 両速度でテスト
        speeds = [70, 98]
        all_results = {}
        
        for speed in speeds:
            print(f"\n🚀 HIGH_SPEED_BASE = {speed} テスト")
            result = self.test_motor_stability(
                base_speed=speed,
                name=f"BASE_SPEED_{speed}",
                duration=5.0
            )
            all_results[speed] = result
            
            if speed < max(speeds):  # 最後でなければ
                input("次の速度テストに進むにはEnterを押してください...")
        
        # 最終比較
        self.final_speed_comparison(all_results)
    
    def final_speed_comparison(self, results):
        """70 vs 98 最終比較"""
        print("\n" + "=" * 60)
        print("🏆 70 vs 98 Pure Motor安定性比較")
        print("=" * 60)
        
        for speed, result in results.items():
            if result:
                print(f"HIGH_SPEED_BASE = {speed}:")
                print(f"   総合安定性: {result['stability_score']:.2f}")
                print(f"   位置安定性: {result['position_stability']:.1f}")
                print(f"   直進性: {result['straightness']:.1f}")
                print()
        
        # 勝者決定
        winner = None
        if len(results) == 2 and all(results.values()):
            speed_70_score = results[70]['stability_score']
            speed_98_score = results[98]['stability_score']
            
            if speed_70_score < speed_98_score:
                winner = 70
                print(f"🥇 勝者: HIGH_SPEED_BASE = 70")
                print(f"   安定性スコア: {speed_70_score:.2f} vs {speed_98_score:.2f}")
            elif speed_98_score < speed_70_score:
                winner = 98
                print(f"🥇 勝者: HIGH_SPEED_BASE = 98")
                print(f"   安定性スコア: {speed_98_score:.2f} vs {speed_70_score:.2f}")
            else:
                print("🤝 引き分け！")
                
            print()
            print("📝 結論:")
            if winner == 98:
                print("✅ HIGH_SPEED_BASE=98の優位性が確認されました")
                print("   より高速でありながら安定性も良好")
            elif winner == 70:
                print("🤔 HIGH_SPEED_BASE=70の方が安定性が高い結果")
                print("   低速の安定性優位が確認されました")
            else:
                print("🤝 両速度とも同等の安定性")

def main():
    tester = Pure5SecMotorTest()
    try:
        tester.run_dual_speed_test()
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
