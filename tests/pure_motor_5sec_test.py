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
        
    def test_pid_setting(self, kp, kd, limits, name, base_speed, duration=5.0):
        """PID設定テスト（外乱あり）"""
        print(f"\n=== {name} (速度{base_speed}) PIDテスト ===")
        print(f"設定: Kp={kp}, Kd={kd}, limits={limits}")
        print("車体を平らな場所にセットして準備...")
        input("準備完了したらEnterを押してください...")
        
        # PID設定
        from nnspike.utils import PIDController
        pid = PIDController(
            Kp=kp,
            Ki=0,
            Kd=kd,
            setpoint=0,
            output_limits=limits
        )
        
        # データ収集用
        positions = []
        corrections = []
        
        print(f"🚗 {name} 5秒PIDテスト開始！")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 微小外乱（ライン追従をシミュレート）
                elapsed = time.time() - start_time
                if elapsed > 1:  # 1秒後から外乱
                    disturbance = random.uniform(-0.02, 0.02)  # ラジアン
                else:
                    disturbance = 0
                
                # PID制御
                steering_correction = pid.update(disturbance)
                corrections.append(abs(steering_correction))
                
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
                
                # 位置データ記録
                status = self.et.get_spike_status()
                if status and status.motors:
                    left_pos = status.motors["A"].relative_position
                    right_pos = status.motors["B"].relative_position
                    positions.append((left_pos, right_pos))
                
                time.sleep(0.02)  # 50Hz
        
        except KeyboardInterrupt:
            print("テスト中断")
        finally:
            self.et.brake()
            time.sleep(0.5)
        
        # 結果分析
        result = self.analyze_pid_results(name, base_speed, positions, corrections)
        return result
    
    def analyze_pid_results(self, name, base_speed, positions, corrections):
        """PID結果分析"""
        if not positions or not corrections:
            print(f"❌ {name}: データ不足")
            return None
        
        # PID制御安定性
        avg_correction = sum(corrections) / len(corrections)
        max_correction = max(corrections)
        
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
        
        print(f"\n📊 {name} (速度{base_speed}) PID結果:")
        print(f"   平均制御量: {avg_correction:.3f}")
        print(f"   最大制御量: {max_correction:.3f}")
        print(f"   位置安定性: {position_stability:.1f}")
        print(f"   左右差平均: {avg_diff:.1f}")
        
        # PID総合スコア（小さいほど良い）
        pid_score = avg_correction * 1000 + position_stability * 0.1 + avg_diff * 0.01
        print(f"   PID総合スコア: {pid_score:.2f} (小さいほど良い)")
        
        return {
            'name': name,
            'speed': base_speed,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'position_stability': position_stability,
            'avg_diff': avg_diff,
            'pid_score': pid_score
        }
    
    def test_motor_stability(self, name, base_speed, duration=3.0):
        """基本モーター安定性テスト"""
        print(f"\n⚙️ {name} (速度{base_speed}) テスト開始...")
        
        positions = []
        disturbances = []
        
        try:
            # ETRobotをインポート
            from nnspike.unit.etrobot import ETRobot
            robot = ETRobot()
            
            # 計測開始
            duration_int = int(duration)
            for second in range(duration_int):
                print(f"   {second+1}秒...")
                
                # モーター速度設定（左右同期）
                robot.set_motor_speed(base_speed, base_speed)
                
                # 1秒間動作
                time.sleep(1.0)
                
                # モーター停止
                robot.set_motor_speed(0, 0)
                
                # 位置取得（相対位置）
                relative_pos = robot.retrieve_motors_relative_position()
                positions.append((relative_pos, relative_pos))  # 簡易的に同じ値を使用
                
                time.sleep(0.1)
            
            print(f"   ✅ {name} 完了")
            return positions, disturbances
            
        except Exception as e:
            print(f"   ❌ {name} エラー: {e}")
            return [], []
    
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
