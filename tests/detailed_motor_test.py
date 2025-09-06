#!/usr/bin/env python3
"""
Detailed Motor Test - 車輪浮上状態での詳細モーターデータ取得

モーター位置、パワー、速度を詳細ディスプレイして実測値を分析
"""

import time
import sys
import os
from typing import List, Dict

# パスを追加してパッケージをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

class DetailedMotorTest:
    """詳細モーターテスト"""
    
    def __init__(self):
        self.et = ETRobot()
        self.base_speed = HIGH_SPEED_BASE  # 98
        
    def display_motor_status(self, status, elapsed_time):
        """モーター状態の詳細表示"""
        motor_a = status.motors["A"]
        motor_b = status.motors["B"]
        
        print(f"Time: {elapsed_time:6.2f}s | "
              f"A_pos: {motor_a.position:6d} | A_rel: {motor_a.relative_position:6d} | "
              f"A_spd: {motor_a.speed:4d} | A_pwr: {motor_a.power:4d} | "
              f"B_pos: {motor_b.position:6d} | B_rel: {motor_b.relative_position:6d} | "
              f"B_spd: {motor_b.speed:4d} | B_pwr: {motor_b.power:4d} | "
              f"SpeedDiff: {abs(motor_a.speed - motor_b.speed):3d}")
    
    def test_fixed_speed(self, speed: int, duration: float = 5.0):
        """固定速度テスト"""
        print(f"\n=== 固定速度テスト: {speed} ===")
        print("Time     | A_pos  | A_rel  | A_spd | A_pwr | B_pos  | B_rel  | B_spd | B_pwr | Diff")
        print("-" * 90)
        
        data_log = []
        
        # モーター開始
        self.et.set_motor_forward_speed(speed, speed)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            status = self.et.get_spike_status()
            
            # データ表示
            self.display_motor_status(status, elapsed)
            
            # データ記録
            motor_a = status.motors["A"]
            motor_b = status.motors["B"]
            data_log.append({
                'time': elapsed,
                'motor_a_position': motor_a.position,
                'motor_a_relative_position': motor_a.relative_position,
                'motor_a_speed': motor_a.speed,
                'motor_a_power': motor_a.power,
                'motor_b_position': motor_b.position,
                'motor_b_relative_position': motor_b.relative_position,
                'motor_b_speed': motor_b.speed,
                'motor_b_power': motor_b.power,
                'speed_diff': abs(motor_a.speed - motor_b.speed)
            })
            
            time.sleep(0.1)  # 10Hz
        
        self.et.brake()
        time.sleep(1.0)
        
        return data_log
    
    def test_speed_range(self, speeds: List[int], duration: float = 3.0):
        """速度範囲テスト"""
        print(f"\n=== 速度範囲テスト: {speeds} ===")
        
        all_data = {}
        
        for speed in speeds:
            print(f"\n--- 速度 {speed} のテスト ---")
            data = self.test_fixed_speed(speed, duration)
            all_data[speed] = data
            
            # 統計表示
            self.display_speed_statistics(speed, data)
        
        return all_data
    
    def display_speed_statistics(self, speed: int, data: List[Dict]):
        """速度統計表示"""
        if not data:
            return
        
        # 統計計算
        a_speeds = [d['motor_a_speed'] for d in data if d['motor_a_speed'] is not None]
        b_speeds = [d['motor_b_speed'] for d in data if d['motor_b_speed'] is not None]
        a_powers = [d['motor_a_power'] for d in data if d['motor_a_power'] is not None]
        b_powers = [d['motor_b_power'] for d in data if d['motor_b_power'] is not None]
        speed_diffs = [d['speed_diff'] for d in data]
        
        if a_speeds and b_speeds:
            avg_a_speed = sum(a_speeds) / len(a_speeds)
            avg_b_speed = sum(b_speeds) / len(b_speeds)
            avg_a_power = sum(a_powers) / len(a_powers) if a_powers else 0
            avg_b_power = sum(b_powers) / len(b_powers) if b_powers else 0
            avg_diff = sum(speed_diffs) / len(speed_diffs)
            max_diff = max(speed_diffs)
            
            efficiency_a = avg_a_speed / speed if speed > 0 else 0
            efficiency_b = avg_b_speed / speed if speed > 0 else 0
            
            print(f"\n📊 速度{speed}の統計:")
            print(f"  指令値: {speed}")
            print(f"  実測A: 平均{avg_a_speed:.1f} (効率{efficiency_a:.3f}) パワー{avg_a_power:.1f}")
            print(f"  実測B: 平均{avg_b_speed:.1f} (効率{efficiency_b:.3f}) パワー{avg_b_power:.1f}")
            print(f"  左右差: 平均{avg_diff:.1f}, 最大{max_diff}")
            print(f"  サンプル数: {len(data)}")
    
    def test_high_speed_exploration(self):
        """高速域探索テスト"""
        print("=== 高速域探索テスト ===")
        print("98を基準として、より高速域での安定性を確認")
        print("車体を浮上させた状態で実行してください")
        
        input("準備ができたらEnterキーを押してください...")
        
        # テスト速度範囲
        test_speeds = [95, 98, 100, 102, 105]
        
        print(f"\nテスト速度: {test_speeds}")
        print("各速度で3秒間のテストを実行します")
        
        results = self.test_speed_range(test_speeds, duration=3.0)
        
        # 総合分析
        self.analyze_speed_range_results(results)
        
        return results
    
    def analyze_speed_range_results(self, results: Dict):
        """速度範囲結果の総合分析"""
        print(f"\n" + "="*80)
        print("📊 高速域探索結果 - 総合分析")
        print("="*80)
        
        print(f"\n🎯 速度別性能比較:")
        print(f"{'速度':<6} {'実測A':<8} {'実測B':<8} {'効率A':<8} {'効率B':<8} {'平均差':<8} {'最大差':<8}")
        print("-" * 70)
        
        best_speed = None
        best_efficiency = None
        best_stability = None
        
        for speed, data in sorted(results.items()):
            if not data:
                continue
                
            a_speeds = [d['motor_a_speed'] for d in data if d['motor_a_speed'] is not None]
            b_speeds = [d['motor_b_speed'] for d in data if d['motor_b_speed'] is not None]
            speed_diffs = [d['speed_diff'] for d in data]
            
            if a_speeds and b_speeds:
                avg_a = sum(a_speeds) / len(a_speeds)
                avg_b = sum(b_speeds) / len(b_speeds)
                avg_diff = sum(speed_diffs) / len(speed_diffs)
                max_diff = max(speed_diffs)
                
                eff_a = avg_a / speed if speed > 0 else 0
                eff_b = avg_b / speed if speed > 0 else 0
                avg_eff = (eff_a + eff_b) / 2
                
                print(f"{speed:<6} {avg_a:<8.1f} {avg_b:<8.1f} {eff_a:<8.3f} {eff_b:<8.3f} {avg_diff:<8.1f} {max_diff:<8}")
                
                # ベスト記録更新
                if best_speed is None or avg_a + avg_b > best_speed[1]:
                    best_speed = (speed, avg_a + avg_b)
                
                if best_efficiency is None or avg_eff > best_efficiency[1]:
                    best_efficiency = (speed, avg_eff)
                
                if best_stability is None or avg_diff < best_stability[1]:
                    best_stability = (speed, avg_diff)
        
        # ベスト結果表示
        print(f"\n🏆 分析結果:")
        if best_speed:
            print(f"  最高実測速度: {best_speed[0]} (合計{best_speed[1]:.1f})")
        if best_efficiency:
            print(f"  最高効率: {best_efficiency[0]} (効率{best_efficiency[1]:.3f})")
        if best_stability:
            print(f"  最安定: {best_stability[0]} (平均差{best_stability[1]:.1f})")
        
        # 推奨事項
        print(f"\n💡 推奨事項:")
        current_speed = 98
        
        if best_speed and best_speed[0] > current_speed:
            print(f"  🚀 高速化余地あり: {current_speed} → {best_speed[0]} への変更を検討")
        else:
            print(f"  ✅ 現在の{current_speed}が最適: 変更不要")
        
        if best_stability and best_stability[1] < 5:
            print(f"  🎯 安定性良好: 左右差{best_stability[1]:.1f}は十分制御範囲内")
        elif best_stability:
            print(f"  ⚠️  安定性要改善: 左右差{best_stability[1]:.1f}、PID調整検討")

def main():
    tester = DetailedMotorTest()
    
    try:
        tester.test_high_speed_exploration()
        print("\n✅ 詳細モーターテスト完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        tester.et.stop()

if __name__ == "__main__":
    main()
