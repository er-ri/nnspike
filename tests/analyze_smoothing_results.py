#!/usr/bin/env python3
"""
Motor Smoothing Analysis - テスト結果の詳細分析

motor_smoothing_test_20250906_131145.json の結果を分析し、
最適なスムージングパラメータを決定します。
"""

import json
import sys
import os

def analyze_motor_smoothing_results():
    """モータースムージング結果の詳細分析"""
    
    # データ読み込み
    results = {
        "baseline": {
            "total_samples": 249,
            "avg_speed_diff": 184.49799196787149,
            "max_speed_diff": 222,
            "std_speed_diff": 40.05048472361738,
            "avg_motor_a_speed": -92.38152610441767,
            "avg_motor_b_speed": 92.11646586345381,
            "motor_a_variation": 20.019622936017647,
            "motor_b_variation": 20.061359754549297
        },
        "step_response": {
            "total_samples": 299,
            "avg_speed_diff": 121.18060200668896,
            "max_speed_diff": 203,
            "std_speed_diff": 73.52729136501162,
            "avg_motor_a_speed": -60.625418060200666,
            "avg_motor_b_speed": 60.55518394648829,
            "motor_a_variation": 36.76900717543138,
            "motor_b_variation": 36.77930915741788
        },
        "ramp_up": {
            "total_samples": 99,
            "avg_speed_diff": 49.92929292929293,
            "max_speed_diff": 134,
            "std_speed_diff": 45.87788861511244,
            "avg_motor_a_speed": -25.353535353535353,
            "avg_motor_b_speed": 24.575757575757574,
            "motor_a_variation": 23.457000665704896,
            "motor_b_variation": 22.43634574500959
        },
        "smoothed_light": {
            "total_samples": 80,
            "avg_speed_diff": 42.9375,
            "max_speed_diff": 90,
            "std_speed_diff": 31.80571708420526,
            "avg_motor_a_speed": -20.925,
            "avg_motor_b_speed": 22.0125,
            "motor_a_variation": 15.025526802618359,
            "motor_b_variation": 16.924623474300265
        },
        "smoothed_heavy": {
            "total_samples": 80,
            "avg_speed_diff": 80.1875,
            "max_speed_diff": 154,
            "std_speed_diff": 49.81995272853456,
            "avg_motor_a_speed": -38.6125,
            "avg_motor_b_speed": 41.575,
            "motor_a_variation": 23.50948838881437,
            "motor_b_variation": 26.590554618824868
        }
    }
    
    print("=== モータースムージング詳細分析 ===")
    print()
    
    # ベースラインとの比較
    baseline = results["baseline"]
    print("📊 ベースライン（スムージングなし）との比較:")
    print(f"   平均速度差: {baseline['avg_speed_diff']:.1f}")
    print(f"   標準偏差: {baseline['std_speed_diff']:.1f}")
    print(f"   最大差: {baseline['max_speed_diff']}")
    print()
    
    # 各手法の改善効果
    print("🎯 各手法の改善効果:")
    
    methods = [
        ("ramp_up", "ランプアップ"),
        ("smoothed_light", "軽めスムージング"),
        ("smoothed_heavy", "強めスムージング")
    ]
    
    for method_key, method_name in methods:
        method = results[method_key]
        
        # 改善率計算
        speed_diff_improvement = (baseline['avg_speed_diff'] - method['avg_speed_diff']) / baseline['avg_speed_diff'] * 100
        std_improvement = (baseline['std_speed_diff'] - method['std_speed_diff']) / baseline['std_speed_diff'] * 100
        max_improvement = (baseline['max_speed_diff'] - method['max_speed_diff']) / baseline['max_speed_diff'] * 100
        
        print(f"   {method_name}:")
        print(f"     平均速度差: {method['avg_speed_diff']:.1f} (改善: {speed_diff_improvement:+.1f}%)")
        print(f"     標準偏差: {method['std_speed_diff']:.1f} (改善: {std_improvement:+.1f}%)")
        print(f"     最大差: {method['max_speed_diff']} (改善: {max_improvement:+.1f}%)")
        print()
    
    # 実用性評価
    print("⚡ 実用性評価:")
    print()
    
    # smoothed_lightの詳細分析
    light = results["smoothed_light"]
    print("🏆 最優秀「軽めスムージング」の特徴:")
    print(f"   ✅ 速度差標準偏差: {light['std_speed_diff']:.1f} (20.6%改善)")
    print(f"   ✅ 最大速度差: {light['max_speed_diff']} (59.5%改善)")
    print(f"   ✅ モーター変動: A={light['motor_a_variation']:.1f}, B={light['motor_b_variation']:.1f}")
    print(f"   ✅ 実装複雑度: 低（smoothing_factor=0.3）")
    print()
    
    # 課題と注意点
    print("⚠️  注意すべき点:")
    print("   1. 全ての結果で左右モーターの極性が反転")
    print("      → 車体浮上時の配線/設定要確認")
    print("   2. ベースライン速度差が184と異常に大きい")
    print("      → HIGH_SPEED_BASE=98での調整が必要")
    print("   3. スムージング強度は適度に（0.3程度）")
    print("      → 過度なスムージングは逆効果")
    print()
    
    # 実装推奨事項
    print("🚀 実装推奨事項:")
    print()
    print("1. **スムージング係数**: 0.3")
    print("   - 軽めのスムージングが最適")
    print("   - PID制御への影響を最小化")
    print()
    print("2. **実装方式**: 指数移動平均")
    print("   ```python")
    print("   current_speed += (target_speed - current_speed) * 0.3")
    print("   ```")
    print()
    print("3. **適用場面**:")
    print("   - HIGH_SPEED_AVOIDモードでの急激な速度変化")
    print("   - ステアリング補正による左右差の平滑化")
    print("   - ライン追従時の振動抑制")
    print()
    print("4. **期待効果**:")
    print("   - 速度差変動の20%削減")
    print("   - 最大速度差の60%削減")
    print("   - より安定したライン追従")
    print()
    
    # 次のステップ
    print("📋 次のステップ:")
    print("1. run_manual.pyにスムージング機能を実装")
    print("2. 実際のライン追従での効果検証")
    print("3. PID制御パラメータの微調整")
    print("4. バッテリー電圧変動への対応")

def generate_smoothing_implementation():
    """スムージング実装コードの生成"""
    
    print("\n" + "="*60)
    print("🔧 実装コード例")
    print("="*60)
    
    code = '''
class MotorSpeedSmoother:
    """モーター速度スムージングクラス"""
    
    def __init__(self, smoothing_factor: float = 0.3):
        self.smoothing_factor = smoothing_factor
        self.current_left = 0.0
        self.current_right = 0.0
        self.enabled = True
    
    def smooth_speeds(self, target_left: int, target_right: int) -> tuple[int, int]:
        """速度をスムージング"""
        if not self.enabled:
            return target_left, target_right
        
        # 指数移動平均でスムージング
        self.current_left += (target_left - self.current_left) * self.smoothing_factor
        self.current_right += (target_right - self.current_right) * self.smoothing_factor
        
        return int(self.current_left), int(self.current_right)
    
    def reset(self):
        """スムージング状態をリセット"""
        self.current_left = 0.0
        self.current_right = 0.0
    
    def set_enabled(self, enabled: bool):
        """スムージングの有効/無効を切り替え"""
        self.enabled = enabled

# run_manual.pyでの使用例:
# スムージング初期化
smoother = MotorSpeedSmoother(smoothing_factor=0.3)

# HIGH_SPEED_AVOIDモード内で使用
if mode == Mode.HIGH_SPEED_AVOID:
    # 通常の制御
    target_x, (left_speed, right_speed, current_base_speed), mode = unpack_action_result(action_chain.high_speed_avoid(frame))
    
    # スムージング適用
    if left_speed is not None and right_speed is not None:
        left_speed, right_speed = smoother.smooth_speeds(left_speed, right_speed)
    '''
    
    print(code)

if __name__ == "__main__":
    analyze_motor_smoothing_results()
    generate_smoothing_implementation()
