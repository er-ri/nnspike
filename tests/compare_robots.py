#!/usr/bin/env python3
"""
機体間モーター性能比較プログラム

2台の機体のモーター性能を比較し、どちらが優れているかを科学的に判定します。
個体差、バッテリー状態、モーター特性などを詳細分析します。
"""

import sys
import time
import csv
import statistics
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

class RobotPerformanceComparator:
    def __init__(self):
        self.et = ETRobot()
        self.test_results = []
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.robot_id = ""
        
    def identify_robot(self):
        """機体を識別"""
        print("=" * 50)
        print("機体識別")
        print("=" * 50)
        
        # 自動識別を試みる
        try:
            status = self.et.get_spike_status()
            battery_voltage = getattr(status, 'battery_voltage', None)
            
            if battery_voltage:
                print(f"検出されたバッテリー電圧: {battery_voltage:.2f}V")
            
            # 機体名を手動入力
            self.robot_id = input("機体名を入力してください (例: Robot-A, Robot-B): ").strip()
            
            if not self.robot_id:
                self.robot_id = f"Robot-{datetime.now().strftime('%H%M')}"
                
            print(f"機体識別: {self.robot_id}")
            
        except Exception as e:
            print(f"自動識別失敗: {e}")
            self.robot_id = input("機体名を入力してください: ").strip()
        
        return self.robot_id
    
    def comprehensive_motor_test(self, duration=10):
        """包括的モーター性能テスト"""
        print(f"\n=== {self.robot_id} - 包括的モーター性能テスト ===")
        print(f"テスト時間: {duration}秒")
        print("車輪を浮かした状態で実行してください")
        
        input("準備ができたらEnterキーを押してください...")
        
        # テスト速度リスト
        test_speeds = [
            (80, "中速基準"),
            (90, "準高速"),
            (98, "最適設定"),
            (100, "SPIKE推奨"),
            (110, "高設定")
        ]
        
        robot_results = []
        
        for i, (speed, desc) in enumerate(test_speeds, 1):
            print(f"\n[{i}/{len(test_speeds)}] {desc} (設定値: {speed})")
            
            try:
                result = self.test_single_speed_for_comparison(speed, duration, desc)
                if result:
                    result['robot_id'] = self.robot_id
                    result['test_order'] = i
                    robot_results.append(result)
                    
                    # 即座に重要指標を表示
                    print(f"  結果: 実速度={result['avg_speed']:.1f}, 安定性={result['stability_score']:.4f}")
                
                # 休憩
                if i < len(test_speeds):
                    print("3秒休憩...")
                    time.sleep(3)
                    
            except KeyboardInterrupt:
                print("\nテスト中断")
                break
            except Exception as e:
                print(f"エラー: {e}")
                continue
        
        # 機体の総合スコアを計算
        if robot_results:
            self.calculate_robot_summary(robot_results)
            
        return robot_results
    
    def test_single_speed_for_comparison(self, test_speed, duration, description):
        """比較用の単一速度テスト"""
        print(f"  測定中... ({duration}秒)")
        
        # データ収集
        speed_data_a = []
        speed_data_b = []
        power_data_a = []
        power_data_b = []
        battery_voltages = []
        
        # 安定化期間
        stabilization_frames = 25  # 0.5秒
        test_frames = int(50 * duration)
        
        start_time = time.time()
        
        for frame in range(test_frames):
            try:
                self.et.set_motor_forward_speed(left_speed=test_speed, right_speed=test_speed)
                
                status = self.et.get_spike_status()
                motor_a = status.motors.get("A")
                motor_b = status.motors.get("B")
                
                # 安定化期間後のデータのみ記録
                if frame >= stabilization_frames and motor_a and motor_b:
                    speed_a = abs(motor_a.speed) if motor_a.speed is not None else 0
                    speed_b = abs(motor_b.speed) if motor_b.speed is not None else 0
                    power_a = abs(motor_a.power) if motor_a.power is not None else 0
                    power_b = abs(motor_b.power) if motor_b.power is not None else 0
                    
                    speed_data_a.append(speed_a)
                    speed_data_b.append(speed_b)
                    power_data_a.append(power_a)
                    power_data_b.append(power_b)
                    
                    # バッテリー電圧も記録
                    if hasattr(status, 'battery_voltage') and status.battery_voltage:
                        battery_voltages.append(status.battery_voltage)
                
                # 進捗表示（簡易版）
                if frame % 100 == 0:
                    elapsed = frame // 50
                    print(f"    {elapsed}/{duration}秒...", end='\r')
                
                time.sleep(0.02)  # 50fps
                
            except Exception as e:
                print(f"  測定エラー: {e}")
                break
        
        self.et.brake()
        print()  # 改行
        
        # 結果計算
        if len(speed_data_a) >= 50:  # 最低1秒分のデータ
            # 基本統計
            avg_speed_a = statistics.mean(speed_data_a)
            avg_speed_b = statistics.mean(speed_data_b)
            avg_power_a = statistics.mean(power_data_a)
            avg_power_b = statistics.mean(power_data_b)
            
            # 安定性指標
            std_speed_a = statistics.stdev(speed_data_a) if len(speed_data_a) > 1 else 0
            std_speed_b = statistics.stdev(speed_data_b) if len(speed_data_b) > 1 else 0
            
            cv_speed_a = std_speed_a / avg_speed_a if avg_speed_a > 0 else 0
            cv_speed_b = std_speed_b / avg_speed_b if avg_speed_b > 0 else 0
            
            # 総合値
            avg_speed = (avg_speed_a + avg_speed_b) / 2
            avg_power = (avg_power_a + avg_power_b) / 2
            avg_cv = (cv_speed_a + cv_speed_b) / 2
            
            # 性能指標
            achievement_rate = avg_speed / test_speed if test_speed > 0 else 0
            stability_score = avg_cv
            efficiency_score = avg_speed / avg_power if avg_power > 0 else 0
            
            # バッテリー状態
            avg_battery = statistics.mean(battery_voltages) if battery_voltages else 0
            
            result = {
                'test_speed': test_speed,
                'description': description,
                'avg_speed_a': avg_speed_a,
                'avg_speed_b': avg_speed_b,
                'avg_speed': avg_speed,
                'avg_power_a': avg_power_a,
                'avg_power_b': avg_power_b,
                'avg_power': avg_power,
                'std_speed_a': std_speed_a,
                'std_speed_b': std_speed_b,
                'cv_speed_a': cv_speed_a,
                'cv_speed_b': cv_speed_b,
                'avg_cv': avg_cv,
                'achievement_rate': achievement_rate,
                'stability_score': stability_score,
                'efficiency_score': efficiency_score,
                'avg_battery': avg_battery,
                'samples': len(speed_data_a),
                'timestamp': time.strftime("%H:%M:%S")
            }
            
            return result
        else:
            print("  十分なデータが取得できませんでした")
            return None
    
    def calculate_robot_summary(self, results):
        """機体の総合性能サマリーを計算"""
        if not results:
            return
            
        # 各指標の平均を計算
        avg_speeds = [r['avg_speed'] for r in results]
        stability_scores = [r['stability_score'] for r in results]
        efficiency_scores = [r['efficiency_score'] for r in results]
        achievement_rates = [r['achievement_rate'] for r in results]
        
        summary = {
            'robot_id': self.robot_id,
            'total_avg_speed': statistics.mean(avg_speeds),
            'total_stability': statistics.mean(stability_scores),
            'total_efficiency': statistics.mean(efficiency_scores),
            'total_achievement': statistics.mean(achievement_rates),
            'speed_consistency': statistics.stdev(avg_speeds) if len(avg_speeds) > 1 else 0,
            'battery_voltage': results[0]['avg_battery'] if results else 0,
            'test_count': len(results),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 総合スコア計算 (各項目を正規化して合計)
        # 高いほど良い: 速度、効率、達成率
        # 低いほど良い: 安定性スコア、速度一貫性
        performance_score = (
            summary['total_avg_speed'] * 0.3 +  # 30%
            (1 / (summary['total_stability'] + 0.001)) * 0.25 +  # 25%
            summary['total_efficiency'] * 0.2 +  # 20%
            summary['total_achievement'] * 100 * 0.15 +  # 15%
            (1 / (summary['speed_consistency'] + 0.001)) * 0.1  # 10%
        )
        
        summary['performance_score'] = performance_score
        
        print(f"\n📊 {self.robot_id} 総合性能サマリー:")
        print(f"  平均実速度: {summary['total_avg_speed']:.1f}")
        print(f"  安定性: {summary['total_stability']:.4f} (低いほど良い)")
        print(f"  効率性: {summary['total_efficiency']:.2f}")
        print(f"  達成率: {summary['total_achievement']:.3f} ({summary['total_achievement']*100:.1f}%)")
        print(f"  速度一貫性: ±{summary['speed_consistency']:.1f}")
        print(f"  バッテリー: {summary['battery_voltage']:.2f}V")
        print(f"  総合スコア: {performance_score:.1f}")
        
        # グローバルリザルトに追加
        self.test_results.extend(results)
        
        return summary
    
    def save_comparison_results(self):
        """比較結果をCSVに保存"""
        if not self.test_results:
            return
            
        csv_filename = f"storage/robot_comparison_{self.timestamp}.csv"
        Path(csv_filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            if self.test_results:
                writer = csv.DictWriter(f, fieldnames=self.test_results[0].keys())
                writer.writeheader()
                writer.writerows(self.test_results)
        
        print(f"\n比較結果を保存しました: {csv_filename}")
    
    def cleanup(self):
        """クリーンアップ"""
        self.et.brake()
        self.et.stop()

def compare_two_robots():
    """2台の機体を比較するメイン関数"""
    print("=" * 60)
    print("機体間モーター性能比較プログラム")
    print("=" * 60)
    
    robot_summaries = []
    
    # 1台目のテスト
    print("\n🤖 1台目の機体をテストします")
    comparator1 = RobotPerformanceComparator()
    
    try:
        robot1_id = comparator1.identify_robot()
        results1 = comparator1.comprehensive_motor_test(duration=8)
        summary1 = comparator1.calculate_robot_summary(results1)
        if summary1:
            robot_summaries.append(summary1)
        comparator1.save_comparison_results()
        
    except Exception as e:
        print(f"1台目のテストでエラー: {e}")
    finally:
        comparator1.cleanup()
    
    # 機体交換の指示
    print("\n" + "=" * 60)
    print("機体を交換してください")
    print("=" * 60)
    input("2台目の機体の準備ができたらEnterキーを押してください...")
    
    # 2台目のテスト
    print("\n🤖 2台目の機体をテストします")
    comparator2 = RobotPerformanceComparator()
    
    try:
        robot2_id = comparator2.identify_robot()
        results2 = comparator2.comprehensive_motor_test(duration=8)
        summary2 = comparator2.calculate_robot_summary(results2)
        if summary2:
            robot_summaries.append(summary2)
        comparator2.save_comparison_results()
        
    except Exception as e:
        print(f"2台目のテストでエラー: {e}")
    finally:
        comparator2.cleanup()
    
    # 比較分析
    if len(robot_summaries) == 2:
        analyze_robot_comparison(robot_summaries)
    else:
        print("⚠️ 2台の完全なテストデータが揃いませんでした")

def analyze_robot_comparison(summaries):
    """2台の機体の比較分析"""
    print("\n" + "=" * 80)
    print("🏆 機体間性能比較結果")
    print("=" * 80)
    
    robot1, robot2 = summaries[0], summaries[1]
    
    # 比較表
    print(f"\n{'項目':>12} {'':>15} {robot1['robot_id']:>15} {robot2['robot_id']:>15} {'優勢':>8}")
    print("-" * 80)
    
    # 各項目の比較
    comparisons = [
        ('平均速度', 'total_avg_speed', '%.1f', True),
        ('安定性', 'total_stability', '%.4f', False),
        ('効率性', 'total_efficiency', '%.2f', True),
        ('達成率', 'total_achievement', '%.3f', True),
        ('一貫性', 'speed_consistency', '%.1f', False),
        ('バッテリー', 'battery_voltage', '%.2fV', True),
        ('総合スコア', 'performance_score', '%.1f', True)
    ]
    
    wins = {robot1['robot_id']: 0, robot2['robot_id']: 0}
    
    for name, key, fmt, higher_better in comparisons:
        val1 = robot1[key]
        val2 = robot2[key]
        
        if higher_better:
            winner = robot1['robot_id'] if val1 > val2 else robot2['robot_id']
        else:
            winner = robot1['robot_id'] if val1 < val2 else robot2['robot_id']
        
        wins[winner] += 1
        
        print(f"{name:>12}: {fmt % val1:>15} {fmt % val2:>15} {winner:>8}")
    
    # 総合判定
    print("\n" + "=" * 80)
    print("🎯 総合判定")
    print("=" * 80)
    
    overall_winner = max(wins, key=wins.get)
    winner_score = max(robot1['performance_score'], robot2['performance_score'])
    score_diff = abs(robot1['performance_score'] - robot2['performance_score'])
    
    print(f"🏆 優勝: {overall_winner}")
    print(f"   勝利項目: {wins[overall_winner]}/7項目")
    print(f"   総合スコア: {winner_score:.1f}")
    print(f"   性能差: {score_diff:.1f}ポイント")
    
    # 性能差の評価
    if score_diff < 5:
        print("   評価: 性能差は僅差")
    elif score_diff < 15:
        print("   評価: 明確な性能差あり")
    else:
        print("   評価: 大きな性能差あり")
    
    # 推奨事項
    print(f"\n💡 推奨事項:")
    
    # バッテリー状態の確認
    battery_diff = abs(robot1['battery_voltage'] - robot2['battery_voltage'])
    if battery_diff > 0.5:
        low_battery_robot = robot1['robot_id'] if robot1['battery_voltage'] < robot2['battery_voltage'] else robot2['robot_id']
        print(f"   {low_battery_robot}のバッテリー状態を確認してください (電圧差: {battery_diff:.2f}V)")
    
    # 安定性の確認
    unstable_robot = robot1 if robot1['total_stability'] > robot2['total_stability'] else robot2
    if unstable_robot['total_stability'] > 0.02:
        print(f"   {unstable_robot['robot_id']}の機械的状態を確認してください (安定性: {unstable_robot['total_stability']:.4f})")
    
    # 効率性の確認
    inefficient_robot = robot1 if robot1['total_efficiency'] < robot2['total_efficiency'] else robot2
    efficiency_diff = abs(robot1['total_efficiency'] - robot2['total_efficiency'])
    if efficiency_diff > 0.2:
        print(f"   {inefficient_robot['robot_id']}のモーター効率を確認してください")

def main():
    """メイン関数"""
    print("機体間モーター性能比較プログラム")
    print("1: 2台比較テスト (推奨)")
    print("2: 単体テスト")
    
    choice = input("選択してください (1-2): ").strip()
    
    if choice == "1":
        compare_two_robots()
    elif choice == "2":
        comparator = RobotPerformanceComparator()
        try:
            comparator.identify_robot()
            results = comparator.comprehensive_motor_test(duration=10)
            comparator.save_comparison_results()
        finally:
            comparator.cleanup()
    else:
        print("無効な選択です")

if __name__ == "__main__":
    main()
