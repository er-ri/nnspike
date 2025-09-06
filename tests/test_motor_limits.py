#!/usr/bin/env python3
"""
モーター性能限界テストプログラム

車輪を浮かした状態で様々な設定値でのモーター性能を測定します。
run_manual.pyやconstants.pyを変更せずに独立してテストできます。
"""

import sys
import time
import csv
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from nnspike.unit import ETRobot
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE

class MotorPerformanceTester:
    def __init__(self):
        self.et = ETRobot()
        self.test_results = []
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    def test_single_speed(self, test_speed, duration=10, description=""):
        """単一速度でのテスト"""
        print(f"\n=== Testing Speed: {test_speed} ({description}) ===")
        
        # データ収集用リスト
        speed_data_a = []
        speed_data_b = []
        power_data_a = []
        power_data_b = []
        
        # テスト開始
        test_frames = int(50 * duration)  # 50fps × duration秒
        
        print(f"Starting {duration}秒間のテスト...")
        start_time = time.time()
        
        for frame in range(test_frames):
            # モーター制御
            try:
                self.et.set_motor_forward_speed(left_speed=test_speed, right_speed=test_speed)
                
                # ステータス取得
                status = self.et.get_spike_status()
                
                # データ記録
                motor_a = status.motors.get("A")
                motor_b = status.motors.get("B")
                
                if motor_a and motor_b:
                    speed_data_a.append(abs(motor_a.speed) if motor_a.speed is not None else 0)
                    speed_data_b.append(abs(motor_b.speed) if motor_b.speed is not None else 0)
                    power_data_a.append(abs(motor_a.power) if motor_a.power is not None else 0)
                    power_data_b.append(abs(motor_b.power) if motor_b.power is not None else 0)
                
                # 進捗表示
                if frame % 50 == 0:  # 1秒ごと
                    elapsed = frame // 50
                    print(f"  {elapsed}/{duration}秒経過... (Speed: A={speed_data_a[-1] if speed_data_a else 0}, B={speed_data_b[-1] if speed_data_b else 0})")
                
                # フレームレート調整
                time.sleep(0.02)  # 50fps
                
            except Exception as e:
                print(f"  エラー発生: {e}")
                break
        
        # モーター停止
        self.et.brake()
        
        # 結果計算
        if speed_data_a and speed_data_b:
            result = {
                'test_speed': test_speed,
                'description': description,
                'duration': duration,
                'frames_collected': len(speed_data_a),
                'avg_speed_a': sum(speed_data_a) / len(speed_data_a),
                'avg_speed_b': sum(speed_data_b) / len(speed_data_b),
                'avg_power_a': sum(power_data_a) / len(power_data_a),
                'avg_power_b': sum(power_data_b) / len(power_data_b),
                'max_speed_a': max(speed_data_a),
                'max_speed_b': max(speed_data_b),
                'max_power_a': max(power_data_a),
                'max_power_b': max(power_data_b),
                'speed_std_a': (sum([(x - sum(speed_data_a)/len(speed_data_a))**2 for x in speed_data_a]) / len(speed_data_a))**0.5,
                'speed_std_b': (sum([(x - sum(speed_data_b)/len(speed_data_b))**2 for x in speed_data_b]) / len(speed_data_b))**0.5,
                'achievement_rate_a': (sum(speed_data_a) / len(speed_data_a)) / test_speed if test_speed > 0 else 0,
                'achievement_rate_b': (sum(speed_data_b) / len(speed_data_b)) / test_speed if test_speed > 0 else 0,
                'timestamp': time.strftime("%H:%M:%S")
            }
            
            self.test_results.append(result)
            
            # 結果表示
            print(f"結果:")
            print(f"  Motor A: 平均速度={result['avg_speed_a']:.1f}, 平均パワー={result['avg_power_a']:.1f}, 達成率={result['achievement_rate_a']:.3f}")
            print(f"  Motor B: 平均速度={result['avg_speed_b']:.1f}, 平均パワー={result['avg_power_b']:.1f}, 達成率={result['achievement_rate_b']:.3f}")
            print(f"  変動: Speed A±{result['speed_std_a']:.1f}, Speed B±{result['speed_std_b']:.1f}")
            
            return result
        else:
            print("データが取得できませんでした")
            return None
    
    def run_comprehensive_test(self):
        """包括的なモーター性能テスト"""
        print("=" * 60)
        print("モーター性能限界テスト開始")
        print("=" * 60)
        print("注意: 車輪を浮かした状態でテストしてください")
        
        input("準備ができたらEnterキーを押してください...")
        
        # テスト速度リスト
        test_speeds = [
            (50, "低速基準"),
            (BASE_SPEED, f"BASE_SPEED({BASE_SPEED})"),
            (HIGH_SPEED_BASE, f"HIGH_SPEED_BASE({HIGH_SPEED_BASE})"),
            (80, "中速"),
            (90, "準高速"),
            (100, "SPIKE推奨上限"),
            (110, "上限超過1"),
            (120, "上限超過2"),
            (140, "上限超過3"),
            (160, "上限超過4"),
            (180, "上限超過5"),
            (200, "高設定値"),
            (220, "超高設定値"),
            (255, "最大設定値")
        ]
        
        print(f"\n{len(test_speeds)}種類の速度設定でテストします")
        
        for i, (speed, desc) in enumerate(test_speeds, 1):
            print(f"\n[{i}/{len(test_speeds)}] {desc}のテスト")
            
            try:
                result = self.test_single_speed(speed, duration=8, description=desc)
                
                # 危険な兆候をチェック
                if result:
                    if result['avg_power_a'] > 95 or result['avg_power_b'] > 95:
                        print("⚠️  警告: パワー使用率が95%を超えました")
                    
                    if result['speed_std_a'] > 30 or result['speed_std_b'] > 30:
                        print("⚠️  警告: 速度変動が大きくなっています")
                
                # 次のテストまで休憩
                if i < len(test_speeds):
                    print("3秒間休憩...")
                    time.sleep(3)
                    
            except KeyboardInterrupt:
                print("\n\nテスト中断")
                break
            except Exception as e:
                print(f"テストエラー: {e}")
                continue
        
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """結果をCSVファイルに保存"""
        if not self.test_results:
            return
            
        csv_filename = f"storage/motor_test_results_{self.timestamp}.csv"
        
        # ディレクトリ作成
        Path(csv_filename).parent.mkdir(parents=True, exist_ok=True)
        
        # CSV保存
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            if self.test_results:
                writer = csv.DictWriter(f, fieldnames=self.test_results[0].keys())
                writer.writeheader()
                writer.writerows(self.test_results)
        
        print(f"\n結果を保存しました: {csv_filename}")
    
    def analyze_results(self):
        """結果分析とレポート生成"""
        if not self.test_results:
            return
            
        print("\n" + "=" * 60)
        print("テスト結果分析")
        print("=" * 60)
        
        print(f"\n{'設定値':>6} {'説明':>12} {'実速度A':>8} {'実速度B':>8} {'実パワーA':>9} {'実パワーB':>9} {'達成率A':>7} {'達成率B':>7}")
        print("-" * 80)
        
        for result in self.test_results:
            print(f"{result['test_speed']:>6d} {result['description']:>12} "
                  f"{result['avg_speed_a']:>8.1f} {result['avg_speed_b']:>8.1f} "
                  f"{result['avg_power_a']:>9.1f} {result['avg_power_b']:>9.1f} "
                  f"{result['achievement_rate_a']:>7.3f} {result['achievement_rate_b']:>7.3f}")
        
        # 重要な発見
        print(f"\n重要な発見:")
        
        # 最大性能
        max_speed = max([max(r['avg_speed_a'], r['avg_speed_b']) for r in self.test_results])
        max_power = max([max(r['avg_power_a'], r['avg_power_b']) for r in self.test_results])
        print(f"• 最大実速度: {max_speed:.1f}")
        print(f"• 最大パワー: {max_power:.1f}")
        
        # 100設定時の性能
        result_100 = next((r for r in self.test_results if r['test_speed'] == 100), None)
        if result_100:
            avg_achievement = (result_100['achievement_rate_a'] + result_100['achievement_rate_b']) / 2
            print(f"• 設定100時の達成率: {avg_achievement:.3f} ({avg_achievement*100:.1f}%)")
        
        # 実用的な推奨値
        high_performance_results = [r for r in self.test_results if 
                                  max(r['avg_power_a'], r['avg_power_b']) < 90 and
                                  min(r['achievement_rate_a'], r['achievement_rate_b']) > 0.8]
        
        if high_performance_results:
            best_result = max(high_performance_results, 
                            key=lambda x: (x['avg_speed_a'] + x['avg_speed_b']) / 2)
            print(f"• 推奨設定値: {best_result['test_speed']} ({best_result['description']})")
            print(f"  - 実速度: {(best_result['avg_speed_a'] + best_result['avg_speed_b'])/2:.1f}")
            print(f"  - パワー使用率: {(best_result['avg_power_a'] + best_result['avg_power_b'])/2:.1f}%")
    
    def run_quick_test(self, speed=100):
        """クイックテスト（1つの速度のみ）"""
        print("=" * 40)
        print(f"クイックテスト (設定値: {speed})")
        print("=" * 40)
        print("注意: 車輪を浮かした状態でテストしてください")
        
        input("準備ができたらEnterキーを押してください...")
        
        result = self.test_single_speed(speed, duration=5, description="クイックテスト")
        
        if result:
            self.test_results = [result]
            self.save_results()
            
            print(f"\nクイックテスト完了!")
            print(f"設定{speed} → 実速度A:{result['avg_speed_a']:.1f}, 実速度B:{result['avg_speed_b']:.1f}")
            print(f"パワーA:{result['avg_power_a']:.1f}%, パワーB:{result['avg_power_b']:.1f}%")
    
    def cleanup(self):
        """クリーンアップ"""
        self.et.brake()
        self.et.stop()

def main():
    """メイン関数"""
    tester = MotorPerformanceTester()
    
    try:
        print("モーター性能テストプログラム")
        print("1: 包括的テスト (全速度域)")
        print("2: クイックテスト (設定値100)")
        print("3: カスタムテスト")
        
        choice = input("\n選択してください (1-3): ").strip()
        
        if choice == "1":
            tester.run_comprehensive_test()
        elif choice == "2":
            tester.run_quick_test(100)
        elif choice == "3":
            try:
                speed = int(input("テスト速度を入力してください (10-255): "))
                if 10 <= speed <= 255:
                    tester.run_quick_test(speed)
                else:
                    print("無効な速度です")
            except ValueError:
                print("数値を入力してください")
        else:
            print("無効な選択です")
            
    except KeyboardInterrupt:
        print("\n\nテスト中断")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
