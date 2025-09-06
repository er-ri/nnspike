#!/usr/bin/env python3
"""
モーター最適化分析 - 実測データに基づく科学的評価
"""

class MotorOptimizationAnalysis:
    """実測データに基づくモーター性能最適化分析"""
    
    def __init__(self):
        # 実測データ (2025年9月6日 車体浮上テスト結果)
        self.test_results = {
            95:  {'speed_a': -86.2, 'speed_b': 86.7, 'efficiency': 0.910, 'avg_diff': 172.9, 'max_diff': 220},
            98:  {'speed_a': -87.8, 'speed_b': 87.8, 'efficiency': 0.896, 'avg_diff': 175.6, 'max_diff': 211},
            100: {'speed_a': -90.7, 'speed_b': 90.8, 'efficiency': 0.908, 'avg_diff': 181.5, 'max_diff': 230},
            102: {'speed_a': -90.6, 'speed_b': 90.6, 'efficiency': 0.888, 'avg_diff': 181.1, 'max_diff': 230},
            105: {'speed_a': -89.4, 'speed_b': 89.6, 'efficiency': 0.854, 'avg_diff': 179.0, 'max_diff': 221}
        }
        
        self.current_setting = 98
        
    def analyze_performance_metrics(self):
        """性能指標の詳細分析"""
        print("🔬 モーター性能指標分析")
        print("=" * 60)
        
        for speed, data in self.test_results.items():
            actual_speed = abs(data['speed_a'])
            efficiency = data['efficiency']
            stability = 1.0 / data['avg_diff'] * 1000  # 安定性指標 (逆数)
            max_deviation = data['max_diff']
            
            # 総合スコア計算 (速度40%, 効率30%, 安定性20%, 最大偏差10%)
            performance_score = (
                actual_speed * 0.4 +
                efficiency * 100 * 0.3 +
                stability * 0.2 -
                max_deviation * 0.1
            )
            
            status = "⭐ 現在設定" if speed == self.current_setting else ""
            
            print(f"速度設定 {speed:3d}: 実測{actual_speed:5.1f} 効率{efficiency:.3f} "
                  f"安定性{stability:.2f} 最大偏差{max_deviation:3.0f} "
                  f"総合{performance_score:.1f} {status}")
    
    def analyze_speed_efficiency_curve(self):
        """速度-効率曲線分析"""
        print("\n📈 速度-効率特性曲線分析")
        print("=" * 60)
        
        speeds = list(self.test_results.keys())
        efficiencies = [self.test_results[s]['efficiency'] for s in speeds]
        
        # 最適効率点の検出
        max_eff_idx = efficiencies.index(max(efficiencies))
        optimal_speed = speeds[max_eff_idx]
        optimal_efficiency = efficiencies[max_eff_idx]
        
        print(f"📊 効率特性:")
        print(f"  最高効率: {optimal_efficiency:.3f} @ 速度{optimal_speed}")
        
        # 効率劣化点の検出
        efficiency_threshold = 0.90
        degraded_speeds = [s for s in speeds if self.test_results[s]['efficiency'] < efficiency_threshold]
        
        if degraded_speeds:
            print(f"  効率劣化: 速度{degraded_speeds} で効率{efficiency_threshold:.2f}未満")
        
        # 98の位置評価
        current_efficiency = self.test_results[self.current_setting]['efficiency']
        efficiency_rank = sorted(efficiencies, reverse=True).index(current_efficiency) + 1
        
        print(f"  現在設定98: 効率{current_efficiency:.3f} (順位{efficiency_rank}/5)")
    
    def analyze_stability_characteristics(self):
        """安定性特性分析"""
        print("\n⚖️ 安定性特性分析")
        print("=" * 60)
        
        avg_diffs = [self.test_results[s]['avg_diff'] for s in self.test_results.keys()]
        max_diffs = [self.test_results[s]['max_diff'] for s in self.test_results.keys()]
        
        min_avg_diff = min(avg_diffs)
        min_max_diff = min(max_diffs)
        
        most_stable_avg = list(self.test_results.keys())[avg_diffs.index(min_avg_diff)]
        most_stable_max = list(self.test_results.keys())[max_diffs.index(min_max_diff)]
        
        print(f"📊 安定性指標:")
        print(f"  最安定平均差: {min_avg_diff:.1f} @ 速度{most_stable_avg}")
        print(f"  最小最大差: {min_max_diff} @ 速度{most_stable_max}")
        
        # 98の安定性評価
        current_avg_diff = self.test_results[self.current_setting]['avg_diff']
        current_max_diff = self.test_results[self.current_setting]['max_diff']
        
        avg_diff_rank = sorted(avg_diffs).index(current_avg_diff) + 1
        max_diff_rank = sorted(max_diffs).index(current_max_diff) + 1
        
        print(f"  現在設定98: 平均差{current_avg_diff:.1f} (順位{avg_diff_rank}/5), "
              f"最大差{current_max_diff} (順位{max_diff_rank}/5)")
    
    def recommend_optimization(self):
        """最適化推奨分析"""
        print("\n💡 最適化推奨事項")
        print("=" * 60)
        
        # 多角的評価による推奨
        recommendations = []
        
        # 1. 現在の98評価
        current_data = self.test_results[self.current_setting]
        current_actual = abs(current_data['speed_a'])
        current_efficiency = current_data['efficiency']
        current_stability = current_data['avg_diff']
        
        print(f"🎯 現在設定98の評価:")
        print(f"  実測速度: {current_actual:.1f} (目標比89.6%)")
        print(f"  効率: {current_efficiency:.3f} (良好)")
        print(f"  安定性: {current_stability:.1f} (中程度)")
        
        # 2. 100の検討
        speed_100_data = self.test_results[100]
        speed_100_actual = abs(speed_100_data['speed_a'])
        speed_100_efficiency = speed_100_data['efficiency']
        speed_100_stability = speed_100_data['avg_diff']
        
        print(f"\n🔍 速度100の可能性:")
        print(f"  実測速度: {speed_100_actual:.1f} (+{speed_100_actual-current_actual:.1f})")
        print(f"  効率: {speed_100_efficiency:.3f} (+{speed_100_efficiency-current_efficiency:.3f})")
        print(f"  安定性: {speed_100_stability:.1f} ({speed_100_stability-current_stability:+.1f})")
        
        # 3. 科学的推奨
        if speed_100_efficiency > current_efficiency and speed_100_actual > current_actual:
            if speed_100_stability < current_stability + 10:  # 安定性悪化が10以内
                recommendations.append("💡 速度100への変更を検討価値あり")
            else:
                recommendations.append("⚠️ 速度100は性能向上するが安定性リスクあり")
        
        # PID制約チェック
        max_left_right_diff = max([data['max_diff'] for data in self.test_results.values()])
        if max_left_right_diff > 200:
            recommendations.append("🔧 PID調整による安定性改善を推奨")
        
        # 最終推奨
        print(f"\n🏆 科学的推奨事項:")
        if not recommendations:
            recommendations.append("✅ 現在の98設定は最適、変更不要")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return recommendations
    
    def generate_summary_report(self):
        """総合レポート生成"""
        print("\n" + "=" * 80)
        print("📋 モーター最適化分析 - 総合レポート")
        print("=" * 80)
        
        self.analyze_performance_metrics()
        self.analyze_speed_efficiency_curve()
        self.analyze_stability_characteristics()
        recommendations = self.recommend_optimization()
        
        print(f"\n📅 分析日時: 2025年9月6日")
        print(f"🔬 テスト条件: 車体浮上状態、3秒間×5速度設定")
        print(f"📊 データ品質: 高品質 (30サンプル/速度)")
        
        return recommendations

if __name__ == "__main__":
    analyzer = MotorOptimizationAnalysis()
    recommendations = analyzer.generate_summary_report()
    
    # 設定変更提案の有無確認
    change_recommended = any("100" in rec for rec in recommendations)
    if change_recommended:
        print(f"\n🤔 HIGH_SPEED_BASE変更提案:")
        print(f"  現在: 98")
        print(f"  提案: 100 (要慎重検討)")
        print(f"  理由: 実速度+2.9, 効率+0.012, 安定性-5.9")
        print(f"  決定: ユーザー判断に委ねる")
    else:
        print(f"\n✅ HIGH_SPEED_BASE=98維持推奨")
        print(f"  科学的根拠: 安定性と性能のバランス最適")
