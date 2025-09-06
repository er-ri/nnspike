#!/usr/bin/env python3
"""
汎用PID設定の制御範囲分析
output_limits=(-2, 2)でどの程度のピクセルずれまで対応できるかを計算
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.constants import HIGH_SPEED_BASE, BASE_SPEED

def analyze_control_range():
    """PID制御範囲とピクセルずれの関係を分析"""
    
    print("汎用PID設定制御範囲分析")
    print("=" * 60)
    
    # 汎用最適設定
    Kp = 0.3
    Kd = 0.3
    max_output = 2.0  # output_limits=(-2, 2)
    
    print(f"🔧 汎用PID設定:")
    print(f"   Kp = {Kp}")
    print(f"   Kd = {Kd}")
    print(f"   output_limits = (-{max_output}, {max_output})")
    print()
    
    # P制御のみで最大ピクセルずれを計算
    max_pixel_error_p = max_output / Kp
    print(f"📊 P制御による最大対応ピクセルずれ:")
    print(f"   最大エラー = output_limit ÷ Kp")
    print(f"   最大エラー = {max_output} ÷ {Kp} = {max_pixel_error_p:.1f} ピクセル")
    print()
    
    # 実際の速度変化を計算
    print(f"⚡ 速度変化への影響:")
    
    # HIGH_SPEED_BASE = 70の場合
    base_speed_70 = 70
    max_speed_change_70 = max_output
    left_speed_70 = base_speed_70 - max_speed_change_70
    right_speed_70 = base_speed_70 + max_speed_change_70
    
    print(f"   HIGH_SPEED_BASE = 70の場合:")
    print(f"     基準速度: {base_speed_70}")
    print(f"     最大制御出力: ±{max_output}")
    print(f"     速度範囲: {left_speed_70} ～ {right_speed_70}")
    print(f"     制御幅: {max_speed_change_70 * 2} (基準の{max_speed_change_70 * 2 / base_speed_70 * 100:.1f}%)")
    print()
    
    # HIGH_SPEED_BASE = 98の場合
    base_speed_98 = 98
    max_speed_change_98 = max_output
    left_speed_98 = base_speed_98 - max_speed_change_98
    right_speed_98 = base_speed_98 + max_speed_change_98
    
    print(f"   HIGH_SPEED_BASE = 98の場合:")
    print(f"     基準速度: {base_speed_98}")
    print(f"     最大制御出力: ±{max_output}")
    print(f"     速度範囲: {left_speed_98} ～ {right_speed_98}")
    print(f"     制御幅: {max_speed_change_98 * 2} (基準の{max_speed_change_98 * 2 / base_speed_98 * 100:.1f}%)")
    print()
    
    # D制御の効果も考慮
    print(f"🎯 D制御の効果:")
    print(f"   Kd = {Kd}により、急激な変化に対して追加制御")
    print(f"   瞬間的なずれ変化率が10ピクセル/秒の場合:")
    d_contribution = Kd * 10
    print(f"   D制御出力 = {Kd} × 10 = {d_contribution}")
    print(f"   → P制御と合わせて、より大きなずれにも対応可能")
    print()
    
    # 実用的な制御範囲
    print(f"📋 実用的制御範囲の評価:")
    print(f"   🟢 安全範囲: 0 ～ {max_pixel_error_p * 0.7:.1f} ピクセル (P制御70%使用)")
    print(f"   🟡 警戒範囲: {max_pixel_error_p * 0.7:.1f} ～ {max_pixel_error_p:.1f} ピクセル (P制御70-100%使用)")
    print(f"   🔴 限界範囲: {max_pixel_error_p:.1f} ピクセル以上 (制御飽和)")
    print()
    
    # 従来設定との比較
    old_kp = 5.0
    old_max_output = 8.0
    old_max_pixel_error = old_max_output / old_kp
    
    print(f"🆚 従来設定との比較:")
    print(f"   従来設定: Kp={old_kp}, output_limits=(-{old_max_output}, {old_max_output})")
    print(f"   従来の最大対応: {old_max_pixel_error:.1f} ピクセル")
    print(f"   汎用設定の最大対応: {max_pixel_error_p:.1f} ピクセル")
    
    if max_pixel_error_p > old_max_pixel_error:
        improvement = max_pixel_error_p - old_max_pixel_error
        print(f"   → 汎用設定の方が {improvement:.1f} ピクセル 広い制御範囲！")
    else:
        reduction = old_max_pixel_error - max_pixel_error_p
        print(f"   → 汎用設定は {reduction:.1f} ピクセル 狭い制御範囲")
        print(f"   ⚠️  しかし、過制御を防ぎ、自然な安定性を活用")
    print()
    
    # 実際のライントレースでの想定
    print(f"🚗 実際のライントレースでの想定:")
    print(f"   通常のライン追従: 0-3 ピクセル程度のずれ")
    print(f"   カーブでのライン追従: 3-5 ピクセル程度のずれ")
    print(f"   急カーブ・障害物回避: 5-8 ピクセル程度のずれ")
    print(f"   緊急回避: 8ピクセル以上のずれ")
    print()
    
    print(f"✅ 結論:")
    print(f"   汎用PID設定は {max_pixel_error_p:.1f} ピクセルまで対応可能")
    print(f"   通常のライントレースには十分な制御範囲")
    print(f"   緊急回避時は制御飽和の可能性あり（設計通り）")

if __name__ == "__main__":
    analyze_control_range()
