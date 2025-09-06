#!/usr/bin/env python3
"""
中間設定案 5秒実走行テスト
Kp=1.0, Kd=1.0, limits=(-4,4) vs 最適設定 Kp=0.3 の実証比較
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot
from nnspike.utils import PIDController
from nnspike.constants import HIGH_SPEED_BASE
import math

class QuickPIDTest:
    def __init__(self):
        self.et = ETRobot()
        
    def test_pid_setting(self, kp, ki, kd, output_limits, base_speed, duration=5.0, description=""):
        """5秒間PID設定テスト"""
        print(f"\n🔬 {description}")
        print(f"   設定: Kp={kp}, Ki={ki}, Kd={kd}, limits={output_limits}")
        print(f"   基準速度: {base_speed}")
        print(f"   テスト時間: {duration}秒")
        
        # PIDコントローラー設定
        pid = PIDController(
            Kp=kp,
            Ki=ki, 
            Kd=kd,
            setpoint=0,
            output_limits=output_limits
        )
        
        # データ収集用
        speed_diffs = []
        control_outputs = []
        positions = []
        
        input(f"🚗 車体をセットして Enterで{description}開始...")
        
        start_time = time.time()
        loop_count = 0
        
        try:
            while time.time() - start_time < duration:
                loop_start = time.time()
                
                # 現在の状態取得
                status = self.et.get_spike_status()
                left_pos = status.motors["A"].relative_position or 0
                right_pos = status.motors["B"].relative_position or 0
                
                # 簡単な位置エラー（直進からのずれ）
                position_error = (left_pos - right_pos) / 1000.0  # mm to relative
                
                # PID制御
                control_output = pid.update(position_error)
                
                # 速度計算
                left_speed = base_speed - control_output
                right_speed = base_speed + control_output
                
                # 制限
                left_speed = max(0, min(255, int(left_speed)))
                right_speed = max(0, min(255, int(right_speed)))
                
                # モーター制御
                self.et.set_motor_forward_speed(left_speed, right_speed)
                
                # データ記録
                speed_diff = abs(left_speed - right_speed)
                speed_diffs.append(speed_diff)
                control_outputs.append(abs(control_output))
                positions.append(abs(position_error))
                
                loop_count += 1
                
                # 50ms周期維持
                loop_time = time.time() - loop_start
                if loop_time < 0.05:
                    time.sleep(0.05 - loop_time)
                    
        except KeyboardInterrupt:
            print("テスト中断")
        finally:
            # 停止
            self.et.brake()
            time.sleep(0.5)
        
        # 結果分析
        if speed_diffs:
            avg_speed_diff = sum(speed_diffs) / len(speed_diffs)
            max_speed_diff = max(speed_diffs)
            avg_control = sum(control_outputs) / len(control_outputs)
            max_control = max(control_outputs)
            avg_position_error = sum(positions) / len(positions)
            max_position_error = max(positions)
            
            # 制御使用率
            max_possible_output = output_limits[1]
            control_usage = avg_control / max_possible_output if max_possible_output > 0 else 0
            
            print(f"📊 {description} 結果:")
            print(f"   ループ数: {loop_count}")
            print(f"   平均速度差: {avg_speed_diff:.1f}")
            print(f"   最大速度差: {max_speed_diff:.1f}")
            print(f"   平均制御出力: {avg_control:.2f}")
            print(f"   最大制御出力: {max_control:.2f}")
            print(f"   制御使用率: {control_usage:.1%}")
            print(f"   平均位置誤差: {avg_position_error:.3f}")
            print(f"   最大位置誤差: {max_position_error:.3f}")
            
            return {
                'avg_speed_diff': avg_speed_diff,
                'max_speed_diff': max_speed_diff,
                'avg_control': avg_control,
                'max_control': max_control,
                'control_usage': control_usage,
                'avg_position_error': avg_position_error,
                'max_position_error': max_position_error,
                'loop_count': loop_count
            }
        else:
            print("❌ データが取得できませんでした")
            return None

    def run_comparison_test(self):
        """比較テスト実行"""
        print("中間設定案 vs 最適設定 実走行比較テスト")
        print("=" * 60)
        print("⚠️ 安全な直線コースで実施してください")
        print("⚠️ 障害物がないことを確認してください")
        
        # モーター位置リセット
        self.et.set_motor_relative_position(0, 0)
        
        # テスト設定
        test_configs = [
            {
                'name': '最適設定 (Kp=0.3)',
                'kp': 0.3, 'ki': 0, 'kd': 0.3,
                'limits': (-2, 2),
                'base_speed': HIGH_SPEED_BASE
            },
            {
                'name': '中間設定案 (Kp=1.0)',
                'kp': 1.0, 'ki': 0, 'kd': 1.0,
                'limits': (-4, 4),
                'base_speed': HIGH_SPEED_BASE
            },
            {
                'name': '従来設定 (Kp=5.0)',
                'kp': 5.0, 'ki': 0, 'kd': 5.0,
                'limits': (-8, 8),
                'base_speed': HIGH_SPEED_BASE
            }
        ]
        
        results = {}
        
        for config in test_configs:
            # モーター位置リセット
            self.et.set_motor_relative_position(0, 0)
            time.sleep(1)
            
            result = self.test_pid_setting(
                kp=config['kp'],
                ki=config['ki'],
                kd=config['kd'],
                output_limits=config['limits'],
                base_speed=config['base_speed'],
                duration=5.0,
                description=config['name']
            )
            
            if result:
                results[config['name']] = result
            
            # 次のテストまで待機
            input("次のテストの準備ができたら Enterを押してください...")
        
        # 比較分析
        self.analyze_comparison(results)
        
    def analyze_comparison(self, results):
        """比較分析"""
        if len(results) < 2:
            print("❌ 比較に十分なデータがありません")
            return
            
        print("\n" + "=" * 60)
        print("📊 実走行比較分析結果")
        print("=" * 60)
        
        # ランキング作成（平均速度差が小さいほど良い）
        ranking = sorted(results.items(), key=lambda x: x[1]['avg_speed_diff'])
        
        print("🏆 実走行安定性ランキング（平均速度差基準）:")
        for i, (name, result) in enumerate(ranking, 1):
            print(f"{i}位: {name}")
            print(f"     平均速度差: {result['avg_speed_diff']:.1f}")
            print(f"     制御使用率: {result['control_usage']:.1%}")
            print(f"     位置誤差: {result['avg_position_error']:.3f}")
        
        # 詳細比較
        print(f"\n📋 詳細比較:")
        for name, result in results.items():
            print(f"【{name}】")
            print(f"  安定性: 平均速度差 {result['avg_speed_diff']:.1f}")
            print(f"  応答性: 最大制御出力 {result['max_control']:.2f}")
            print(f"  効率性: 制御使用率 {result['control_usage']:.1%}")
            print(f"  精度: 位置誤差 {result['avg_position_error']:.3f}")
        
        # 勝者判定
        winner = ranking[0]
        print(f"\n🎯 実走行テスト勝者: {winner[0]}")
        print(f"   理由: 最も安定した走行（速度差{winner[1]['avg_speed_diff']:.1f}）")
        
        # 中間設定の評価
        if '中間設定案 (Kp=1.0)' in results and '最適設定 (Kp=0.3)' in results:
            intermediate = results['中間設定案 (Kp=1.0)']
            optimal = results['最適設定 (Kp=0.3)']
            
            print(f"\n💡 中間設定案の実証評価:")
            if intermediate['avg_speed_diff'] < optimal['avg_speed_diff']:
                print(f"   ✅ 中間設定が実走行では優秀！")
                print(f"   速度差: {intermediate['avg_speed_diff']:.1f} < {optimal['avg_speed_diff']:.1f}")
            else:
                diff = intermediate['avg_speed_diff'] - optimal['avg_speed_diff']
                print(f"   ❌ 最適設定が実走行でも優秀")
                print(f"   速度差: {optimal['avg_speed_diff']:.1f} vs {intermediate['avg_speed_diff']:.1f} (差{diff:.1f})")

if __name__ == "__main__":
    tester = QuickPIDTest()
    try:
        tester.run_comparison_test()
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        tester.et.stop()
