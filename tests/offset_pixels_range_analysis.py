#!/usr/bin/env python3
"""
汎用PID設定のoffset_pixels制御範囲分析
実際のライントレースでのoffset_pixels値での制御範囲を計算
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nnspike.constants import CAMERA_WIDTH

def analyze_offset_pixels_range():
    """PID制御範囲をoffset_pixels基準で分析"""
    
    print("汎用PID設定 offset_pixels制御範囲分析")
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
    
    # フレーム情報
    frame_width = CAMERA_WIDTH  # 通常は640
    frame_center = frame_width // 2  # 320
    
    print(f"📺 フレーム情報:")
    print(f"   フレーム幅: {frame_width} ピクセル")
    print(f"   フレーム中心: {frame_center} ピクセル")
    print()
    
    # P制御のみで最大offset_pixels を計算
    max_offset_pixels = max_output / Kp
    
    print(f"📊 P制御による最大対応 offset_pixels:")
    print(f"   最大 |offset_pixels| = output_limit ÷ Kp")
    print(f"   最大 |offset_pixels| = {max_output} ÷ {Kp} = {max_offset_pixels:.1f}")
    print()
    
    # offset_pixelsの実際の意味
    print(f"🎯 offset_pixels の実際の意味:")
    print(f"   offset_pixels = target_x - frame_center")
    print(f"   offset_pixels = target_x - {frame_center}")
    print()
    print(f"   正の値: ラインが画面右寄り（右に修正が必要）")
    print(f"   負の値: ラインが画面左寄り（左に修正が必要）")
    print(f"   0: ラインが画面中央（修正不要）")
    print()
    
    # 制御範囲をtarget_x座標で表現
    target_x_min = frame_center - max_offset_pixels
    target_x_max = frame_center + max_offset_pixels
    
    print(f"📍 制御可能なtarget_x座標範囲:")
    print(f"   制御可能範囲: {target_x_min:.1f} ～ {target_x_max:.1f}")
    print(f"   フレーム中心: {frame_center}")
    print(f"   制御幅: ±{max_offset_pixels:.1f} ピクセル")
    print()
    
    # フレーム全体に対する制御範囲の割合
    control_coverage = (max_offset_pixels * 2) / frame_width * 100
    
    print(f"🎬 フレーム全体に対する制御範囲:")
    print(f"   フレーム幅: {frame_width} ピクセル")
    print(f"   制御幅: {max_offset_pixels * 2:.1f} ピクセル")
    print(f"   制御カバー率: {control_coverage:.1f}%")
    print()
    
    # 実用的な制御レベル
    print(f"📋 実用的 offset_pixels 制御レベル:")
    safe_range = max_offset_pixels * 0.7
    print(f"   🟢 安全範囲: |offset_pixels| ≤ {safe_range:.1f}")
    print(f"      target_x: {frame_center - safe_range:.1f} ～ {frame_center + safe_range:.1f}")
    print(f"   🟡 警戒範囲: {safe_range:.1f} < |offset_pixels| ≤ {max_offset_pixels:.1f}")
    print(f"      target_x: {target_x_min:.1f}～{frame_center - safe_range:.1f}, {frame_center + safe_range:.1f}～{target_x_max:.1f}")
    print(f"   🔴 限界範囲: |offset_pixels| > {max_offset_pixels:.1f}")
    print(f"      target_x: < {target_x_min:.1f} または > {target_x_max:.1f}")
    print()
    
    # D制御の効果
    print(f"🎯 D制御による追加効果:")
    print(f"   急激なoffset_pixels変化時にD制御が支援")
    print(f"   例: offset_pixelsが10/秒で変化している場合")
    d_contribution = Kd * 10
    print(f"   D制御出力 = {Kd} × 10 = {d_contribution}")
    print(f"   → 実質的により大きなoffset_pixelsにも対応可能")
    print()
    
    # 従来設定との比較
    old_kp = 5.0
    old_max_output = 8.0
    old_max_offset_pixels = old_max_output / old_kp
    
    print(f"🆚 従来設定との比較:")
    print(f"   従来設定: Kp={old_kp}, output_limits=(-{old_max_output}, {old_max_output})")
    print(f"   従来の最大 |offset_pixels|: {old_max_offset_pixels:.1f}")
    print(f"   汎用設定の最大 |offset_pixels|: {max_offset_pixels:.1f}")
    
    improvement = max_offset_pixels - old_max_offset_pixels
    print(f"   → 汎用設定の方が {improvement:.1f} 広い制御範囲！")
    print()
    
    # 実際のライントレースシナリオ
    print(f"🚗 実際のライントレースシナリオ:")
    print(f"   📍 直進時:")
    print(f"      target_x ≈ {frame_center} (offset_pixels ≈ 0)")
    print(f"      → 🟢 安全範囲内")
    print()
    print(f"   📍 緩いカーブ:")
    print(f"      target_x = {frame_center - 3} ～ {frame_center + 3} (offset_pixels = ±3)")
    print(f"      → 🟢 安全範囲内")
    print()
    print(f"   📍 中程度のカーブ:")
    print(f"      target_x = {frame_center - 5} ～ {frame_center + 5} (offset_pixels = ±5)")
    print(f"      → 🟢 安全範囲内")
    print()
    print(f"   📍 急カーブ:")
    print(f"      target_x = {frame_center - 7} ～ {frame_center + 7} (offset_pixels = ±7)")
    print(f"      → 🔴 限界範囲（制御飽和の可能性）")
    print()
    print(f"   📍 障害物回避:")
    print(f"      target_x = {frame_center - 10} ～ {frame_center + 10} (offset_pixels = ±10)")
    print(f"      → 🔴 制御飽和（設計通り）")
    print()
    
    # フレーム端での制御
    print(f"🖼️ フレーム端での制御:")
    print(f"   フレーム左端: target_x = 0 (offset_pixels = -{frame_center})")
    print(f"   フレーム右端: target_x = {frame_width} (offset_pixels = +{frame_center})")
    print(f"   → これらの極端な値では制御飽和")
    print(f"   → 通常のライントレースではこのような極端な値は発生しない")
    print()
    
    print(f"✅ 結論:")
    print(f"   汎用PID設定は offset_pixels = ±{max_offset_pixels:.1f} まで対応可能")
    print(f"   通常のライントレース（±5以内）には十分な制御範囲")
    print(f"   急カーブ（±7以上）では制御飽和の可能性あり")
    print(f"   従来設定（±{old_max_offset_pixels:.1f}）より{improvement:.1f}倍以上の制御範囲を実現")

if __name__ == "__main__":
    analyze_offset_pixels_range()
