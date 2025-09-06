#!/usr/bin/env python3
"""
5秒実走行PID比較テスト
安全な直線中心で最適設定 vs 中間設定を比較
"""

import sys
import os
import time
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.unit import ETRobot, ActionChain
from nnspike.utils import PIDController
from nnspike.constants import HIGH_SPEED_BASE, CAMERA_WIDTH
import cv2

class Quick5SecPIDTest:
    def __init__(self):
        self.et = ETRobot()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def test_pid_setting(self, kp, kd, limits, name, base_speed, duration=5.0):
        """5秒間の安全な直線PIDテスト"""
        print(f"\n=== {name} テスト開始 (速度{base_speed}) ===")
        print(f"設定: Kp={kp}, Kd={kd}, limits={limits}, BASE_SPEED={base_speed}")
        print("車体を直線ライン上にセットして準備...")
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
        
        print(f"🚗 {name} 5秒テスト開始！(速度{base_speed})")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 簡単なライン検出（安全のため中央重視）
                target_x = self.detect_safe_line(frame)
                
                if target_x is not None:
                    # offset_pixels計算
                    frame_center = CAMERA_WIDTH // 2
                    offset_pixels = target_x - frame_center
                    
                    # PID制御
                    theta = math.atan2(offset_pixels, CAMERA_WIDTH)
                    steering_correction = pid.update(theta)
                    
                    # 指定された基準速度を使用
                    left_speed = base_speed - steering_correction
                    right_speed = base_speed + steering_correction
                    
                    # 安全な速度制限
                    left_speed = max(10, min(120, left_speed))
                    right_speed = max(10, min(120, right_speed))
                    
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
        result = self.analyze_results(name, positions, speeds, corrections, base_speed)
        return result
    
    def detect_safe_line(self, frame):
        """安全重視の簡単ライン検出"""
        try:
            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 画面中央部分のみ使用（安全のため）
            h, w = gray.shape
            roi_top = h // 3
            roi_bottom = h * 2 // 3
            roi_left = w // 4
            roi_right = w * 3 // 4
            
            roi = gray[roi_top:roi_bottom, roi_left:roi_right]
            
            # 二値化
            _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
            
            # 重心計算
            moments = cv2.moments(binary)
            if moments["m00"] > 1000:  # 十分な面積がある場合のみ
                cx = int(moments["m10"] / moments["m00"])
                return roi_left + cx
            else:
                # ライン検出失敗時は中央を返す
                return w // 2
                
        except Exception:
            return CAMERA_WIDTH // 2
    
    def analyze_results(self, name, positions, speeds, corrections, base_speed):
        """結果分析"""
        if not positions:
            print(f"❌ {name}: データ不足")
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
        
        # 実際の平均速度
        avg_speed = sum([l + r for l, r in speeds]) / (2 * len(speeds)) if speeds else base_speed
        speed_efficiency = avg_speed / base_speed if base_speed > 0 else 1.0
        
        print(f"\n📊 {name} 結果 (速度{base_speed}):")
        print(f"   平均制御量: {avg_correction:.2f}")
        print(f"   最大制御量: {max_correction:.2f}")
        print(f"   速度差平均: {avg_speed_diff:.2f}")
        print(f"   位置安定性: {position_stability:.1f}")
        print(f"   実際平均速度: {avg_speed:.1f}")
        print(f"   速度効率: {speed_efficiency:.3f}")
        
        # 総合評価
        stability_score = 1.0 / (1.0 + avg_correction + avg_speed_diff * 0.1 + position_stability * 0.001)
        print(f"   総合安定性: {stability_score:.3f}")
        
        return {
            'name': name,
            'base_speed': base_speed,
            'avg_correction': avg_correction,
            'max_correction': max_correction,
            'avg_speed_diff': avg_speed_diff,
            'position_stability': position_stability,
            'avg_speed': avg_speed,
            'speed_efficiency': speed_efficiency,
            'stability_score': stability_score
        }
    
    def run_comparison_test(self):
        """5秒比較テスト実行"""
        print("🧪 5秒実走行PID比較テスト")
        print("=" * 50)
        print("⚠️ 安全のため直線ライン上でテストしてください")
        print("⚠️ 周囲に障害物がないことを確認してください")
        
        # テスト設定
        test_configs = [
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
            },
            {
                'name': '従来設定 (参考)',
                'kp': 5.0,
                'kd': 5.0,
                'limits': (-8, 8)
            }
        ]
        
        results = {}
        
        for config in test_configs:
            result = self.test_pid_setting(
                kp=config['kp'],
                kd=config['kd'],
                limits=config['limits'],
                name=config['name'],
                duration=5.0
            )
            results[config['name']] = result
            
            print(f"\n{config['name']} 完了")
            input("次のテストに進むには車体を再配置してEnterを押してください...")
        
        # 最終比較
        self.final_comparison(results)
    
    def final_comparison(self, results):
        """最終比較結果"""
        print("\n" + "=" * 60)
        print("🏆 5秒実走行テスト最終結果")
        print("=" * 60)
        
        # 各設定の安定性スコアで比較
        scores = []
        for name, (positions, speeds, corrections) in results.items():
            if corrections:
                avg_correction = sum(corrections) / len(corrections)
                stability_score = 1.0 / (1.0 + avg_correction)
                scores.append((name, stability_score, avg_correction))
        
        # スコア順でソート
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print("ランキング:")
        for i, (name, score, correction) in enumerate(scores, 1):
            print(f"{i}位: {name}")
            print(f"     安定性スコア: {score:.3f}")
            print(f"     平均制御量: {correction:.2f}")
            print()
        
        # 勝者発表
        if scores:
            winner = scores[0][0]
            print(f"🥇 勝者: {winner}")
            
            if "最適設定" in winner:
                print("✅ 科学的テストの結果が実走行でも証明されました！")
            elif "中間設定" in winner:
                print("🤔 中間設定が予想外に良い結果でした...")
            else:
                print("😲 予想外の結果です...")

def main():
    tester = Quick5SecPIDTest()
    try:
        tester.run_comparison_test()
    finally:
        tester.et.stop()
        tester.cap.release()

if __name__ == "__main__":
    main()
