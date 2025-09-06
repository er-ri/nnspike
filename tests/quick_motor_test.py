#!/usr/bin/env python3
"""
モーター限界値簡易テスト

車輪を浮かした状態で瞬時にモーター性能をチェック
最小限のコードで素早く結果を確認できます
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

def quick_motor_check(speed=100, duration=5):
    """モーター性能の簡易チェック"""
    et = ETRobot()
    
    try:
        print(f"=== 簡易モーターテスト (設定値: {speed}) ===")
        print("車輪を浮かしてください!")
        input("準備OK? Enterで開始...")
        
        # データ収集
        speeds_a, speeds_b = [], []
        powers_a, powers_b = [], []
        
        print(f"\n{duration}秒間測定中...")
        start_time = time.time()
        
        # テスト実行
        for i in range(duration * 25):  # 25fps
            et.set_motor_forward_speed(left_speed=speed, right_speed=speed)
            
            try:
                status = et.get_spike_status()
                motor_a = status.motors.get("A")
                motor_b = status.motors.get("B")
                
                if motor_a and motor_b:
                    speeds_a.append(abs(motor_a.speed) if motor_a.speed else 0)
                    speeds_b.append(abs(motor_b.speed) if motor_b.speed else 0)
                    powers_a.append(abs(motor_a.power) if motor_a.power else 0)
                    powers_b.append(abs(motor_b.power) if motor_b.power else 0)
            except:
                pass
            
            # 1秒ごとに進捗表示
            if i % 25 == 0:
                elapsed = i // 25
                current_speed = speeds_a[-1] if speeds_a else 0
                print(f"  {elapsed+1}/{duration}秒 - 現在速度: {current_speed:.0f}")
            
            time.sleep(0.04)  # 25fps
        
        # モーター停止
        et.brake()
        
        # 結果計算・表示
        if speeds_a and speeds_b:
            avg_speed_a = sum(speeds_a) / len(speeds_a)
            avg_speed_b = sum(speeds_b) / len(speeds_b)
            avg_power_a = sum(powers_a) / len(powers_a)
            avg_power_b = sum(powers_b) / len(powers_b)
            
            achievement_a = avg_speed_a / speed
            achievement_b = avg_speed_b / speed
            
            print(f"\n=== 結果 ===")
            print(f"設定値: {speed}")
            print(f"Motor A: 実速度={avg_speed_a:.1f}, パワー={avg_power_a:.1f}%, 達成率={achievement_a:.3f}")
            print(f"Motor B: 実速度={avg_speed_b:.1f}, パワー={avg_power_b:.1f}%, 達成率={achievement_b:.3f}")
            print(f"平均達成率: {(achievement_a + achievement_b)/2:.3f} ({(achievement_a + achievement_b)*50:.1f}%)")
            
            # 簡易判定
            avg_achievement = (achievement_a + achievement_b) / 2
            avg_power = (avg_power_a + avg_power_b) / 2
            
            if avg_power > 95:
                print("💡 判定: パワー限界に近い (95%以上)")
            elif avg_achievement > 0.9:
                print("✅ 判定: 良好な性能")
            elif avg_achievement > 0.8:
                print("⚠️ 判定: 性能やや低下")
            else:
                print("❌ 判定: 大幅な性能低下")
                
            return {
                'speed': speed,
                'avg_speed': (avg_speed_a + avg_speed_b) / 2,
                'avg_power': avg_power,
                'achievement': avg_achievement
            }
        else:
            print("❌ データ取得失敗")
            return None
            
    except Exception as e:
        print(f"エラー: {e}")
        return None
    finally:
        et.brake()
        et.stop()

def compare_speeds():
    """複数速度の比較テスト"""
    speeds = [HIGH_SPEED_BASE, 100, 120, 140]
    results = []
    
    print("=== 複数速度比較テスト ===")
    print(f"テスト速度: {speeds}")
    
    for speed in speeds:
        print(f"\n--- 速度 {speed} ---")
        result = quick_motor_check(speed, duration=3)
        if result:
            results.append(result)
        
        if speed != speeds[-1]:  # 最後でなければ休憩
            print("2秒休憩...")
            time.sleep(2)
    
    # 比較結果表示
    if len(results) > 1:
        print(f"\n=== 比較結果 ===")
        print(f"{'設定':>4} {'実速度':>6} {'パワー':>6} {'達成率':>6}")
        print("-" * 26)
        
        for r in results:
            print(f"{r['speed']:>4} {r['avg_speed']:>6.1f} {r['avg_power']:>6.1f} {r['achievement']:>6.3f}")
        
        # ベスト判定
        best = max(results, key=lambda x: x['avg_speed'])
        print(f"\n最高実速度: 設定{best['speed']} → {best['avg_speed']:.1f}")

if __name__ == "__main__":
    print("簡易モーターテスト")
    print("1: 単発テスト")
    print("2: 比較テスト")
    
    choice = input("選択 (1 or 2): ").strip()
    
    if choice == "1":
        try:
            speed = int(input(f"テスト速度 (デフォルト: {HIGH_SPEED_BASE}): ") or HIGH_SPEED_BASE)
            quick_motor_check(speed)
        except ValueError:
            quick_motor_check(HIGH_SPEED_BASE)
    elif choice == "2":
        compare_speeds()
    else:
        # デフォルト
        quick_motor_check(HIGH_SPEED_BASE)
