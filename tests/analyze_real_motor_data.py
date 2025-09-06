#!/usr/bin/env python3
"""
実測モーターデータ分析 - HIGH_SPEED_BASE=98の詳細解析
"""

import statistics
from typing import List, Tuple, Dict

class RealMotorDataAnalyzer:
    """実測モーターデータの詳細分析"""
    
    def __init__(self):
        # 実測データ (HIGH_SPEED_BASE=98)
        self.motor_data = [
            # [motor_a_pos, motor_a_rel, motor_a_speed, motor_a_power, motor_b_pos, motor_b_rel, motor_b_speed, motor_b_power]
            [-109, 0, 0, 0, -168, 0, 0, 0],
            [-109, 0, 0, 0, -168, 0, 0, 0],
            [-111, -2, -1, -16, -166, 0, 1, 16],
            [-111, -2, -1, -16, -166, 0, 1, 16],
            [-157, -47, -30, -99, -106, 60, 40, 99],
            [-157, -47, -30, -99, -106, 60, 40, 99],
            [-157, -47, -30, -99, -106, 60, 40, 99],
            [87, -164, -84, -98, 18, 185, 91, 98],
            [87, -164, -84, -98, 18, 185, 91, 98],
            [-59, -309, -107, -80, 159, 325, 104, 91],
            [-59, -309, -107, -80, 159, 325, 104, 91],
            [167, -443, -99, -74, -72, 455, 99, 85],
            [167, -443, -99, -74, -72, 455, 99, 85],
            [167, -443, -99, -74, -72, 455, 99, 85],
            [47, -563, -87, -77, 50, 577, 88, 91],
            [47, -563, -87, -77, 50, 577, 88, 91],
            [-73, -683, -88, -80, 176, 702, 92, 88],
            [-73, -683, -88, -80, 176, 702, 92, 88],
            [163, -807, -90, -83, -53, 833, 98, 88],
            [163, -807, -90, -83, -53, 833, 98, 88],
            [37, -933, -90, -85, 74, 961, 96, 91],
            [37, -933, -90, -85, 74, 961, 96, 91],
            [-90, -1060, -94, -86, -155, 1092, 97, 85],
            [-90, -1060, -94, -86, -155, 1092, 97, 85],
            [144, -1187, -91, -89, -22, 1224, 98, 82],
            [144, -1187, -91, -89, -22, 1224, 98, 82],
            [144, -1187, -91, -89, -22, 1224, 98, 82],
            [22, -1308, -87, -90, 106, 1352, 95, 76],
            [22, -1308, -87, -90, 106, 1352, 95, 76],
            [-105, -1435, -92, -89, -137, 1470, 89, 80],
            [-105, -1435, -92, -89, -137, 1470, 89, 80],
            [123, -1567, -96, -89, -14, 1593, 91, 83],
            [123, -1567, -96, -89, -14, 1593, 91, 83],
            [-6, -1696, -92, -92, 117, 1724, 97, 85],
            [-6, -1696, -92, -92, 117, 1724, 97, 85],
            [-136, -1826, -94, -91, -109, 1857, 97, 83],
            [-136, -1826, -94, -91, -109, 1857, 97, 83],
            [94, -1956, -96, -88, 20, 1986, 94, 82],
            [94, -1956, -96, -88, 20, 1986, 94, 82],
            [-36, -2086, -95, -89, 143, 2110, 95, 79],
            [-36, -2086, -95, -89, 143, 2110, 95, 79],
            [-36, -2086, -95, -89, 143, 2110, 95, 79],
            [-168, -2218, -96, -89, -91, 2236, 92, 81],
            [-168, -2218, -96, -89, -91, 2236, 92, 81],
            [62, -2348, -96, -89, 33, 2360, 90, 87],
            [62, -2348, -96, -89, 33, 2360, 90, 87],
            [-71, -2481, -98, -86, 163, 2490, 96, 88],
            [-71, -2481, -98, -86, 163, 2490, 96, 88],
            [157, -2613, -97, -82, -67, 2620, 99, 87],
            [157, -2613, -97, -82, -67, 2620, 99, 87],
            [31, -2739, -90, -85, 65, 2751, 96, 89],
            [31, -2739, -90, -85, 65, 2751, 96, 89],
            [-97, -2868, -94, -89, -160, 2886, 99, 89],
            [-97, -2868, -94, -89, -160, 2886, 99, 89],
            [-97, -2868, -94, -89, -160, 2886, 99, 89],
            [130, -3000, -97, -88, -24, 3023, 103, 83],
            [130, -3000, -97, -88, -24, 3023, 103, 83],
            [3, -3127, -90, -91, 108, 3154, 98, 83],
            [3, -3127, -90, -91, 108, 3154, 98, 83],
            [-125, -3255, -92, -93, -129, 3277, 92, 84],
            [-125, -3255, -92, -93, -129, 3277, 92, 84],
            [-125, -3255, -92, -93, -129, 3277, 92, 84]
        ]
        
        # 安定期のデータのみ抽出 (最初の6サンプルを除外)
        self.stable_data = self.motor_data[6:]
    
    def analyze_speed_characteristics(self):
        """速度特性の詳細分析"""
        print("🔬 速度特性詳細分析 (HIGH_SPEED_BASE=98)")
        print("=" * 60)
        
        # 速度データ抽出 (安定期のみ)
        speeds_a = [abs(row[2]) for row in self.stable_data if row[2] != 0]
        speeds_b = [abs(row[6]) for row in self.stable_data if row[6] != 0]
        
        if not speeds_a or not speeds_b:
            print("有効な速度データが見つかりません")
            return
        
        # 統計計算
        avg_speed_a = statistics.mean(speeds_a)
        avg_speed_b = statistics.mean(speeds_b)
        std_speed_a = statistics.stdev(speeds_a) if len(speeds_a) > 1 else 0
        std_speed_b = statistics.stdev(speeds_b) if len(speeds_b) > 1 else 0
        
        print(f"📊 実測速度統計:")
        print(f"  モーターA: 平均{avg_speed_a:.1f} ±{std_speed_a:.1f} (範囲: {min(speeds_a)}-{max(speeds_a)})")
        print(f"  モーターB: 平均{avg_speed_b:.1f} ±{std_speed_b:.1f} (範囲: {min(speeds_b)}-{max(speeds_b)})")
        print(f"  実効効率: A={avg_speed_a/98:.3f}, B={avg_speed_b/98:.3f}")
        print()
        
        # 速度差分析
        speed_diffs = []
        for row in self.stable_data:
            if row[2] != 0 and row[6] != 0:
                diff = abs(abs(row[2]) - abs(row[6]))
                speed_diffs.append(diff)
        
        if speed_diffs:
            avg_diff = statistics.mean(speed_diffs)
            max_diff = max(speed_diffs)
            min_diff = min(speed_diffs)
            std_diff = statistics.stdev(speed_diffs) if len(speed_diffs) > 1 else 0
            
            print(f"⚖️ 左右速度差分析:")
            print(f"  平均差: {avg_diff:.1f} ±{std_diff:.1f}")
            print(f"  最大差: {max_diff} (許容範囲: <10)")
            print(f"  最小差: {min_diff}")
            print(f"  差異率: {avg_diff/avg_speed_a*100:.1f}%")
        
        return {
            'avg_speed_a': avg_speed_a,
            'avg_speed_b': avg_speed_b,
            'avg_diff': avg_diff if 'avg_diff' in locals() else 0,
            'max_diff': max_diff if 'max_diff' in locals() else 0
        }
    
    def analyze_power_efficiency(self):
        """パワー効率分析"""
        print("\n⚡ パワー効率分析")
        print("=" * 60)
        
        # パワーデータ抽出 (安定期のみ)
        powers_a = [abs(row[3]) for row in self.stable_data if row[3] != 0]
        powers_b = [abs(row[7]) for row in self.stable_data if row[7] != 0]
        speeds_a = [abs(row[2]) for row in self.stable_data if row[2] != 0]
        speeds_b = [abs(row[6]) for row in self.stable_data if row[6] != 0]
        
        if not powers_a or not powers_b:
            print("有効なパワーデータが見つかりません")
            return
        
        avg_power_a = statistics.mean(powers_a)
        avg_power_b = statistics.mean(powers_b)
        avg_speed_a = statistics.mean(speeds_a)
        avg_speed_b = statistics.mean(speeds_b)
        
        # 効率計算 (速度/パワー)
        efficiency_a = avg_speed_a / avg_power_a if avg_power_a > 0 else 0
        efficiency_b = avg_speed_b / avg_power_b if avg_power_b > 0 else 0
        
        print(f"📊 パワー統計:")
        print(f"  モーターA: 平均{avg_power_a:.1f} (範囲: {min(powers_a)}-{max(powers_a)})")
        print(f"  モーターB: 平均{avg_power_b:.1f} (範囲: {min(powers_b)}-{max(powers_b)})")
        print(f"  パワー効率: A={efficiency_a:.3f}, B={efficiency_b:.3f} (速度/パワー)")
        print()
        
        # パワーバランス
        power_diff = abs(avg_power_a - avg_power_b)
        power_ratio = avg_power_a / avg_power_b if avg_power_b > 0 else 1
        
        print(f"⚖️ パワーバランス:")
        print(f"  平均差: {power_diff:.1f}")
        print(f"  比率: {power_ratio:.3f} (理想値: 1.000)")
        
        return {
            'avg_power_a': avg_power_a,
            'avg_power_b': avg_power_b,
            'efficiency_a': efficiency_a,
            'efficiency_b': efficiency_b,
            'power_diff': power_diff
        }
    
    def identify_problematic_patterns(self):
        """問題パターンの特定"""
        print("\n🚨 問題パターン特定")
        print("=" * 60)
        
        issues = []
        
        # 1. 大きな速度差の検出
        large_diff_count = 0
        for row in self.stable_data:
            if row[2] != 0 and row[6] != 0:
                diff = abs(abs(row[2]) - abs(row[6]))
                if diff > 10:  # PID制御の限界を超える差
                    large_diff_count += 1
        
        if large_diff_count > 0:
            issues.append(f"⚠️ 大きな速度差: {large_diff_count}回 (PID制御限界超過)")
        
        # 2. パワー異常の検出
        power_spikes = 0
        for row in self.stable_data:
            if abs(row[3]) > 95 or abs(row[7]) > 95:
                power_spikes += 1
        
        if power_spikes > 0:
            issues.append(f"⚠️ パワースパイク: {power_spikes}回 (>95%)")
        
        # 3. 速度変動の検出
        speed_variations_a = []
        speed_variations_b = []
        
        for i in range(1, len(self.stable_data)):
            if self.stable_data[i-1][2] != 0 and self.stable_data[i][2] != 0:
                variation_a = abs(self.stable_data[i][2] - self.stable_data[i-1][2])
                speed_variations_a.append(variation_a)
            
            if self.stable_data[i-1][6] != 0 and self.stable_data[i][6] != 0:
                variation_b = abs(self.stable_data[i][6] - self.stable_data[i-1][6])
                speed_variations_b.append(variation_b)
        
        if speed_variations_a:
            avg_variation_a = statistics.mean(speed_variations_a)
            if avg_variation_a > 5:
                issues.append(f"⚠️ モーターA速度変動大: 平均{avg_variation_a:.1f}")
        
        if speed_variations_b:
            avg_variation_b = statistics.mean(speed_variations_b)
            if avg_variation_b > 5:
                issues.append(f"⚠️ モーターB速度変動大: 平均{avg_variation_b:.1f}")
        
        if not issues:
            issues.append("✅ 重大な問題は検出されませんでした")
        
        for issue in issues:
            print(f"  {issue}")
        
        return issues
    
    def suggest_improvements(self):
        """改善提案"""
        print("\n💡 改善提案")
        print("=" * 60)
        
        # データ分析結果から改善案を導出
        speed_stats = self.analyze_speed_characteristics()
        power_stats = self.analyze_power_efficiency()
        
        suggestions = []
        
        # 1. 速度差に基づく提案
        if 'avg_diff' in speed_stats and speed_stats['avg_diff'] > 8:
            suggestions.append({
                'issue': f"平均速度差 {speed_stats['avg_diff']:.1f} > 8",
                'solution': "PIDパラメータの微調整",
                'priority': "高",
                'implementation': "Kp=5→4, Kd=5→6 で安定性向上"
            })
        
        # 2. 効率に基づく提案
        if 'avg_speed_a' in speed_stats:
            efficiency = speed_stats['avg_speed_a'] / 98
            if efficiency < 0.90:
                suggestions.append({
                    'issue': f"実効効率 {efficiency:.3f} < 0.90",
                    'solution': "モーター出力最適化",
                    'priority': "中",
                    'implementation': "HIGH_SPEED_BASE=100の検討"
                })
        
        # 3. パワーバランスに基づく提案
        if 'power_diff' in power_stats and power_stats['power_diff'] > 5:
            suggestions.append({
                'issue': f"パワー差 {power_stats['power_diff']:.1f} > 5",
                'solution': "モーター個体差補正",
                'priority': "中",
                'implementation': "left_motor_factor, right_motor_factorの調整"
            })
        
        # 4. スムージングに関する提案
        suggestions.append({
            'issue': "98での過度なスムージング",
            'solution': "スムージング係数の最適化",
            'priority': "低",
            'implementation': "smoothing_factor = 0.0 または 0.1-0.2"
        })
        
        if not suggestions:
            suggestions.append({
                'issue': "なし",
                'solution': "現在の設定は良好",
                'priority': "なし",
                'implementation': "HIGH_SPEED_BASE=98を維持"
            })
        
        print("🎯 優先度別改善提案:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. 【{suggestion['priority']}】{suggestion['issue']}")
            print(f"   解決策: {suggestion['solution']}")
            print(f"   実装: {suggestion['implementation']}")
        
        return suggestions
    
    def generate_comprehensive_report(self):
        """総合レポート生成"""
        print("=" * 80)
        print("📋 HIGH_SPEED_BASE=98 実測データ分析レポート")
        print("=" * 80)
        
        speed_stats = self.analyze_speed_characteristics()
        power_stats = self.analyze_power_efficiency()
        issues = self.identify_problematic_patterns()
        suggestions = self.suggest_improvements()
        
        print(f"\n📅 分析日時: 2025年9月6日")
        print(f"🔬 データ品質: 高品質 ({len(self.stable_data)}サンプル)")
        print(f"⚙️ 設定: HIGH_SPEED_BASE=98 (車体浮上状態)")
        
        return {
            'speed_stats': speed_stats,
            'power_stats': power_stats,
            'issues': issues,
            'suggestions': suggestions
        }

if __name__ == "__main__":
    analyzer = RealMotorDataAnalyzer()
    report = analyzer.generate_comprehensive_report()
