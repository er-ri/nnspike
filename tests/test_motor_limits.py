#!/usr/bin/env python3
"""
モーター性能詳細分析プログラム (95-115範囲特化)

車輪を浮かした状態で95～115の細かい設定値でのモーター性能を測定します。
「110が安定」の根拠を科学的に検証し、PID制御への影響も分析します。
run_manual.pyやconstants.pyを変更せずに独立してテストできます。
"""

import sys
import time
import csv
import statistics
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from nnspike.unit import ETRobot
from nnspike.constants import BASE_SPEED, HIGH_SPEED_BASE

class MotorDetailedAnalyzer:
    def __init__(self):
        self.et = ETRobot()
        self.test_results = []
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    def test_single_speed_detailed(self, test_speed, duration=12, description=""):
        """詳細な単一速度テスト - 安定性分析強化版"""
        print(f"\n=== Testing Speed: {test_speed} ({description}) ===")
        
        # データ収集用リスト
        speed_data_a = []
        speed_data_b = []
        power_data_a = []
        power_data_b = []
        time_stamps = []
        
        # 初期安定化期間（最初の1秒は除外）
        stabilization_frames = 50
        
        # テスト開始
        test_frames = int(50 * duration)  # 50fps × duration秒
        
        print(f"Starting {duration}秒間の詳細テスト...")
        print("(最初の1秒は安定化期間として除外)")
        start_time = time.time()
        
        for frame in range(test_frames):
            # モーター制御
            try:
                self.et.set_motor_forward_speed(left_speed=test_speed, right_speed=test_speed)
                
                # ステータス取得
                status = self.et.get_spike_status()
                
                # データ記録 (安定化期間後のみ)
                motor_a = status.motors.get("A")
                motor_b = status.motors.get("B")
                
                if motor_a and motor_b and frame >= stabilization_frames:
                    current_time = time.time() - start_time
                    speed_a = abs(motor_a.speed) if motor_a.speed is not None else 0
                    speed_b = abs(motor_b.speed) if motor_b.speed is not None else 0
                    power_a = abs(motor_a.power) if motor_a.power is not None else 0
                    power_b = abs(motor_b.power) if motor_b.power is not None else 0
                    
                    speed_data_a.append(speed_a)
                    speed_data_b.append(speed_b)
                    power_data_a.append(power_a)
                    power_data_b.append(power_b)
                    time_stamps.append(current_time)
                
                # 進捗表示 (2秒ごと)
                if frame % 100 == 0:  # 2秒ごと
                    elapsed = frame // 50
                    current_speed_a = abs(motor_a.speed) if motor_a and motor_a.speed else 0
                    current_speed_b = abs(motor_b.speed) if motor_b and motor_b.speed else 0
                    print(f"  {elapsed}/{duration}秒... Speed: A={current_speed_a}, B={current_speed_b}")
                
                # フレームレート調整
                time.sleep(0.02)  # 50fps
                
            except Exception as e:
                print(f"  エラー発生: {e}")
                break
        
        # モーター停止
        self.et.brake()
        
        # 詳細結果計算
        if len(speed_data_a) >= 100:  # 最低2秒分のデータが必要
            # 基本統計
            avg_speed_a = statistics.mean(speed_data_a)
            avg_speed_b = statistics.mean(speed_data_b)
            avg_power_a = statistics.mean(power_data_a)
            avg_power_b = statistics.mean(power_data_b)
            
            # 安定性指標
            std_speed_a = statistics.stdev(speed_data_a)
            std_speed_b = statistics.stdev(speed_data_b)
            std_power_a = statistics.stdev(power_data_a)
            std_power_b = statistics.stdev(power_data_b)
            
            # 変動係数 (CV) - 安定性の重要指標
            cv_speed_a = std_speed_a / avg_speed_a if avg_speed_a > 0 else 0
            cv_speed_b = std_speed_b / avg_speed_b if avg_speed_b > 0 else 0
            cv_power_a = std_power_a / avg_power_a if avg_power_a > 0 else 0
            cv_power_b = std_power_b / avg_power_b if avg_power_b > 0 else 0
            
            # 最大変動幅
            max_speed_a = max(speed_data_a)
            min_speed_a = min(speed_data_a)
            max_speed_b = max(speed_data_b)
            min_speed_b = min(speed_data_b)
            
            speed_range_a = max_speed_a - min_speed_a
            speed_range_b = max_speed_b - min_speed_b
            
            # PID制御余裕度計算
            actual_avg_speed = (avg_speed_a + avg_speed_b) / 2
            pid_headroom = test_speed - actual_avg_speed
            pid_headroom_ratio = pid_headroom / test_speed if test_speed > 0 else 0
            
            # 達成率
            achievement_rate_a = avg_speed_a / test_speed if test_speed > 0 else 0
            achievement_rate_b = avg_speed_b / test_speed if test_speed > 0 else 0
            
            result = {
                'test_speed': test_speed,
                'description': description,
                'duration': duration,
                'samples_collected': len(speed_data_a),
                
                # 基本性能
                'avg_speed_a': avg_speed_a,
                'avg_speed_b': avg_speed_b,
                'avg_power_a': avg_power_a,
                'avg_power_b': avg_power_b,
                'achievement_rate_a': achievement_rate_a,
                'achievement_rate_b': achievement_rate_b,
                
                # 安定性指標
                'std_speed_a': std_speed_a,
                'std_speed_b': std_speed_b,
                'cv_speed_a': cv_speed_a,
                'cv_speed_b': cv_speed_b,
                'cv_power_a': cv_power_a,
                'cv_power_b': cv_power_b,
                
                # 変動幅
                'speed_range_a': speed_range_a,
                'speed_range_b': speed_range_b,
                'max_speed_a': max_speed_a,
                'min_speed_a': min_speed_a,
                'max_speed_b': max_speed_b,
                'min_speed_b': min_speed_b,
                
                # PID制御関連
                'pid_headroom': pid_headroom,
                'pid_headroom_ratio': pid_headroom_ratio,
                'actual_avg_speed': actual_avg_speed,
                
                # 総合安定性スコア (低いほど安定)
                'stability_score': (cv_speed_a + cv_speed_b + cv_power_a + cv_power_b) / 4,
                
                'timestamp': time.strftime("%H:%M:%S")
            }
            
            self.test_results.append(result)
            
            # 詳細結果表示
            print(f"詳細結果:")
            print(f"  基本性能:")
            print(f"    Motor A: 速度={avg_speed_a:.1f}±{std_speed_a:.1f}, パワー={avg_power_a:.1f}±{std_power_a:.1f}")
            print(f"    Motor B: 速度={avg_speed_b:.1f}±{std_speed_b:.1f}, パワー={avg_power_b:.1f}±{std_power_b:.1f}")
            print(f"  安定性:")
            print(f"    速度変動係数: A={cv_speed_a:.4f}, B={cv_speed_b:.4f}")
            print(f"    速度変動幅: A={speed_range_a:.1f}, B={speed_range_b:.1f}")
            print(f"    安定性スコア: {result['stability_score']:.4f} (低いほど安定)")
            print(f"  PID制御関連:")
            print(f"    実速度: {actual_avg_speed:.1f}")
            print(f"    PID余裕度: {pid_headroom:.1f} ({pid_headroom_ratio:.3f})")
            
            return result
        else:
            print("十分なデータが取得できませんでした")
            return None
    
    def run_precision_test_95_115(self):
        """95～115の精密テスト - 「110が安定」の検証"""
        print("=" * 60)
        print("精密モーター分析テスト (95-115範囲)")
        print("「110が安定」という仮説を科学的に検証します")
        print("=" * 60)
        print("注意: 車輪を浮かした状態でテストしてください")
        
        input("準備ができたらEnterキーを押してください...")
        
        # 精密テスト速度リスト
        test_speeds = [
            (95, "現在のHIGH_SPEED_BASE"),
            (98, "前回のBASE設定"),
            (100, "SPIKE推奨上限"),
            (102, "微増1"),
            (105, "微増2"),
            (108, "微増3"),
            (110, "仮説最適値"),
            (112, "微増4"),
            (115, "微増5")
        ]
        
        print(f"\n{len(test_speeds)}種類の速度設定で精密テストします")
        print("各テスト12秒間（安定化1秒＋測定11秒）")
        
        for i, (speed, desc) in enumerate(test_speeds, 1):
            print(f"\n[{i}/{len(test_speeds)}] {desc} (設定値: {speed})")
            
            try:
                result = self.test_single_speed_detailed(speed, duration=12, description=desc)
                
                # 即座に重要指標をチェック
                if result:
                    print(f"  💡 重要指標:")
                    print(f"     安定性スコア: {result['stability_score']:.4f}")
                    print(f"     PID余裕度: {result['pid_headroom']:.1f} ({result['pid_headroom_ratio']:.1%})")
                    print(f"     速度変動係数: {(result['cv_speed_a'] + result['cv_speed_b'])/2:.4f}")
                
                # 休憩時間
                if i < len(test_speeds):
                    print("4秒間休憩...")
                    time.sleep(4)
                    
            except KeyboardInterrupt:
                print("\n\nテスト中断")
                break
            except Exception as e:
                print(f"テストエラー: {e}")
                continue
        
        self.save_detailed_results()
        self.analyze_precision_results()
    
    def run_stability_comparison(self):
        """安定性に特化した比較テスト"""
        print("=" * 60)
        print("安定性比較テスト")
        print("110 vs 100 vs 95の安定性を詳細比較")
        print("=" * 60)
        
        test_speeds = [(95, "現在設定"), (100, "SPIKE推奨"), (110, "仮説最適")]
        
        print("各設定を15秒間テスト（安定化2秒＋測定13秒）")
        input("準備ができたらEnterキーを押してください...")
        
        for speed, desc in test_speeds:
            print(f"\n=== {desc} (設定{speed}) の安定性テスト ===")
            result = self.test_single_speed_detailed(speed, duration=15, description=f"{desc}_安定性")
            
            if result:
                print(f"安定性評価:")
                print(f"  総合スコア: {result['stability_score']:.4f}")
                print(f"  速度変動: ±{(result['std_speed_a'] + result['std_speed_b'])/2:.1f}")
                print(f"  PID余裕: {result['pid_headroom']:.1f}")
            
            time.sleep(3)
        
        self.save_detailed_results()
        self.analyze_stability_results()
    
    def save_detailed_results(self):
        """詳細結果をCSVファイルに保存"""
        if not self.test_results:
            return
            
        csv_filename = f"storage/motor_detailed_analysis_{self.timestamp}.csv"
        
        # ディレクトリ作成
        Path(csv_filename).parent.mkdir(parents=True, exist_ok=True)
        
        # CSV保存
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            if self.test_results:
                writer = csv.DictWriter(f, fieldnames=self.test_results[0].keys())
                writer.writeheader()
                writer.writerows(self.test_results)
        
        print(f"\n詳細結果を保存しました: {csv_filename}")
    
    def analyze_precision_results(self):
        """精密テスト結果の分析とレポート生成"""
        if not self.test_results:
            return
            
        print("\n" + "=" * 80)
        print("精密テスト結果分析 - 「110が安定」の検証")
        print("=" * 80)
        
        # テーブルヘッダー
        print(f"\n{'設定':>3} {'説明':>15} {'実速度':>6} {'PID余裕':>7} {'安定スコア':>8} {'速度CV':>7} {'変動幅':>6} {'総合評価':>8}")
        print("-" * 80)
        
        # 各結果の表示と評価
        best_stability = None
        best_performance = None
        best_balance = None
        
        for result in self.test_results:
            # 総合評価計算
            stability_rank = 1 / (result['stability_score'] + 0.0001)  # 低いほど良い
            performance_rank = result['actual_avg_speed']
            pid_rank = result['pid_headroom_ratio'] * 100
            
            # バランススコア (安定性40%, 性能30%, PID余裕30%)
            balance_score = (stability_rank * 0.4 + performance_rank * 0.3 + pid_rank * 0.3)
            
            avg_cv = (result['cv_speed_a'] + result['cv_speed_b']) / 2
            avg_range = (result['speed_range_a'] + result['speed_range_b']) / 2
            
            evaluation = self._evaluate_setting(result)
            
            print(f"{result['test_speed']:>3d} {result['description']:>15} "
                  f"{result['actual_avg_speed']:>6.1f} {result['pid_headroom']:>7.1f} "
                  f"{result['stability_score']:>8.4f} {avg_cv:>7.4f} "
                  f"{avg_range:>6.1f} {evaluation:>8}")
            
            # ベスト候補の更新
            if best_stability is None or result['stability_score'] < best_stability['stability_score']:
                best_stability = result
            if best_performance is None or result['actual_avg_speed'] > best_performance['actual_avg_speed']:
                best_performance = result
            if best_balance is None or balance_score > best_balance.get('balance_score', 0):
                best_balance = result
                best_balance['balance_score'] = balance_score
        
        # 結論
        print(f"\n🎯 検証結果:")
        if best_stability:
            print(f"最高安定性: 設定{best_stability['test_speed']} (スコア: {best_stability['stability_score']:.4f})")
        if best_performance:
            print(f"最高性能: 設定{best_performance['test_speed']} (実速度: {best_performance['actual_avg_speed']:.1f})")
        if best_balance:
            print(f"最適バランス: 設定{best_balance['test_speed']} (総合評価)")
        
        # 110の特別分析
        result_110 = next((r for r in self.test_results if r['test_speed'] == 110), None)
        if result_110:
            print(f"\n🔍 設定110の詳細分析:")
            print(f"  安定性ランキング: {self._get_stability_rank(result_110)}位/{len(self.test_results)}")
            print(f"  性能ランキング: {self._get_performance_rank(result_110)}位/{len(self.test_results)}")
            print(f"  PID余裕度: {result_110['pid_headroom']:.1f} (比率: {result_110['pid_headroom_ratio']:.1%})")
            
            if result_110 == best_stability:
                print("  ✅ 最高安定性を達成")
            if result_110 == best_balance:
                print("  ✅ 最適バランスを達成")
        
        # 実用的推奨
        print(f"\n💡 実用的推奨:")
        
        # 安定性重視
        stable_candidates = [r for r in self.test_results if r['stability_score'] < 0.05]
        if stable_candidates:
            best_stable = max(stable_candidates, key=lambda x: x['actual_avg_speed'])
            print(f"  安定性重視: 設定{best_stable['test_speed']} (安定 + 高性能)")
        
        # PID制御重視
        pid_candidates = [r for r in self.test_results if r['pid_headroom_ratio'] > 0.1]
        if pid_candidates:
            best_pid = max(pid_candidates, key=lambda x: x['actual_avg_speed'])
            print(f"  PID制御重視: 設定{best_pid['test_speed']} (制御余裕 + 性能)")
        
        # 「110が安定」の検証結論
        if result_110:
            is_most_stable = result_110 == best_stability
            has_good_performance = result_110['actual_avg_speed'] >= 95.0
            has_pid_headroom = result_110['pid_headroom_ratio'] >= 0.08
            
            print(f"\n📊 「設定110が最適」の検証結果:")
            print(f"  最高安定性: {'✅' if is_most_stable else '❌'}")
            print(f"  十分な性能 (95以上): {'✅' if has_good_performance else '❌'}")
            print(f"  PID余裕 (8%以上): {'✅' if has_pid_headroom else '❌'}")
            
            if is_most_stable and has_good_performance and has_pid_headroom:
                print("  🎉 結論: 設定110は科学的に最適と証明されました")
            elif is_most_stable:
                print("  ⚠️ 結論: 設定110は最高安定性だが、他の要因を確認してください")
            else:
                print("  ❌ 結論: 設定110が最適という仮説は否定されました")
    
    def analyze_stability_results(self):
        """安定性比較結果の分析"""
        if len(self.test_results) < 3:
            return
            
        print("\n" + "=" * 60)
        print("安定性比較分析結果")
        print("=" * 60)
        
        # 最新の3つの結果を比較 (95, 100, 110)
        recent_results = self.test_results[-3:]
        
        print("設定値別安定性比較:")
        for result in recent_results:
            print(f"\n設定{result['test_speed']} ({result['description']}):")
            print(f"  安定性スコア: {result['stability_score']:.4f}")
            print(f"  速度変動幅: A={result['speed_range_a']:.1f}, B={result['speed_range_b']:.1f}")
            print(f"  変動係数: A={result['cv_speed_a']:.4f}, B={result['cv_speed_b']:.4f}")
            print(f"  PID余裕度: {result['pid_headroom']:.1f} ({result['pid_headroom_ratio']:.1%})")
        
        # 最安定設定の特定
        most_stable = min(recent_results, key=lambda x: x['stability_score'])
        print(f"\n🏆 最安定設定: {most_stable['test_speed']}")
        print(f"   理由: 安定性スコア {most_stable['stability_score']:.4f} が最小")
    
    def _evaluate_setting(self, result):
        """設定値の総合評価"""
        score = result['stability_score']
        if score < 0.03:
            return "優秀"
        elif score < 0.05:
            return "良好"
        elif score < 0.08:
            return "普通"
        else:
            return "不安定"
    
    def _get_stability_rank(self, target_result):
        """安定性ランキングを取得"""
        sorted_results = sorted(self.test_results, key=lambda x: x['stability_score'])
        return sorted_results.index(target_result) + 1
    
    def _get_performance_rank(self, target_result):
        """性能ランキングを取得"""
        sorted_results = sorted(self.test_results, key=lambda x: x['actual_avg_speed'], reverse=True)
        return sorted_results.index(target_result) + 1
    
    def run_quick_verification(self, speed=110):
        """クイック検証テスト"""
        print("=" * 40)
        print(f"クイック検証テスト (設定値: {speed})")
        print("=" * 40)
        print("注意: 車輪を浮かした状態でテストしてください")
        
        input("準備ができたらEnterキーを押してください...")
        
        result = self.test_single_speed_detailed(speed, duration=8, description="クイック検証")
        
        if result:
            self.test_results = [result]
            self.save_detailed_results()
            
            print(f"\nクイック検証完了!")
            print(f"設定{speed} → 実速度:{result['actual_avg_speed']:.1f}")
            print(f"安定性スコア: {result['stability_score']:.4f}")
            print(f"PID余裕度: {result['pid_headroom']:.1f}")
    
    def cleanup(self):
        """クリーンアップ"""
        self.et.brake()
        self.et.stop()

def main():
    """メイン関数"""
    analyzer = MotorDetailedAnalyzer()
    
    try:
        print("モーター性能詳細分析プログラム")
        print("「110が安定」の科学的検証ツール")
        print("1: 精密テスト (95-115範囲, 推奨)")
        print("2: 安定性比較 (95 vs 100 vs 110)")
        print("3: クイック検証 (110のみ)")
        print("4: カスタム検証")
        
        choice = input("\n選択してください (1-4): ").strip()
        
        if choice == "1":
            analyzer.run_precision_test_95_115()
        elif choice == "2":
            analyzer.run_stability_comparison()
        elif choice == "3":
            analyzer.run_quick_verification(110)
        elif choice == "4":
            try:
                speed = int(input("テスト速度を入力してください (95-115推奨): "))
                if 50 <= speed <= 255:
                    analyzer.run_quick_verification(speed)
                else:
                    print("無効な速度です")
            except ValueError:
                print("数値を入力してください")
        else:
            print("無効な選択です")
            
    except KeyboardInterrupt:
        print("\n\nテスト中断")
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()
