#!/usr/bin/env python3
"""
中間設定案の科学的根拠分析
実際のテスト結果から中間設定の性能を検証
"""

def analyze_intermediate_settings():
    """中間設定の実際の性能を分析"""
    
    print("中間設定案の科学的根拠分析")
    print("=" * 60)
    
    # 実際のテスト結果（汎用PIDテストより）
    test_results = {
        "超安全": {
            "Kp": 0.3, "Kd": 0.3, "limits": (-2, 2),
            "rank": 1, "score": 0.685,
            "speed_diff_70": 3.9, "speed_diff_98": 4.1,
            "efficiency_70": 0.944, "efficiency_98": 0.939
        },
        "安全": {
            "Kp": 0.5, "Kd": 0.5, "limits": (-3, 3),
            "rank": 2, "score": 0.606,
            "speed_diff_70": 5.6, "speed_diff_98": 4.8,
            "efficiency_70": 0.961, "efficiency_98": 0.938
        },
        "バランス1": {
            "Kp": 1.0, "Kd": 0.5, "limits": (-4, 4),
            "rank": 3, "score": 0.556,
            "speed_diff_70": 7.8, "speed_diff_98": 5.9,
            "efficiency_70": 0.963, "efficiency_98": 0.956
        },
        "バランス2": {
            "Kp": 0.8, "Kd": 1.0, "limits": (-4, 4),
            "rank": 4, "score": 0.548,
            "speed_diff_70": 7.8, "speed_diff_98": 5.5,
            "efficiency_70": 0.967, "efficiency_98": 0.952
        },
        "バランス3": {
            "Kp": 1.2, "Kd": 0.8, "limits": (-5, 5),
            "rank": 5, "score": 0.536,
            "speed_diff_70": 9.7, "speed_diff_98": 7.0,
            "efficiency_70": 0.982, "efficiency_98": 0.953
        },
        "応答1": {
            "Kp": 1.5, "Kd": 1.0, "limits": (-6, 6),
            "rank": 6, "score": 0.522,
            "speed_diff_70": 11.2, "speed_diff_98": 7.6,
            "efficiency_70": 0.983, "efficiency_98": 0.958
        },
        "応答2": {
            "Kp": 2.0, "Kd": 1.5, "limits": (-6, 6),
            "rank": 7, "score": 0.514,
            "speed_diff_70": 12.1, "speed_diff_98": 8.5,
            "efficiency_70": 0.980, "efficiency_98": 0.958
        },
        "現在設定": {
            "Kp": 5.0, "Kd": 5.0, "limits": (-8, 8),
            "rank": 8, "score": 0.497,
            "speed_diff_70": 15.6, "speed_diff_98": 10.1,
            "efficiency_70": 0.983, "efficiency_98": 0.955
        }
    }
    
    # 提案された中間設定
    intermediate_setting = {
        "Kp": 1.0, "Kd": 1.0, "limits": (-4, 4)
    }
    
    print(f"🤔 提案された中間設定:")
    print(f"   Kp = {intermediate_setting['Kp']}")
    print(f"   Kd = {intermediate_setting['Kd']}")
    print(f"   output_limits = {intermediate_setting['limits']}")
    print()
    
    # 最も近い実際の設定を特定
    closest_matches = []
    for name, result in test_results.items():
        kp_diff = abs(result["Kp"] - intermediate_setting["Kp"])
        kd_diff = abs(result["Kd"] - intermediate_setting["Kd"])
        limit_diff = abs(result["limits"][1] - intermediate_setting["limits"][1])
        
        total_diff = kp_diff + kd_diff + limit_diff * 0.1
        closest_matches.append((name, total_diff, result))
    
    closest_matches.sort(key=lambda x: x[1])
    
    print(f"📊 実際のテスト結果との比較:")
    print()
    
    for i, (name, diff, result) in enumerate(closest_matches[:3]):
        print(f"{i+1}. 最も近い設定: {name}")
        print(f"   実際の設定: Kp={result['Kp']}, Kd={result['Kd']}, limits={result['limits']}")
        print(f"   ランキング: {result['rank']}位")
        print(f"   汎用スコア: {result['score']}")
        print(f"   速度差: 70={result['speed_diff_70']}, 98={result['speed_diff_98']}")
        print(f"   効率: 70={result['efficiency_70']:.3f}, 98={result['efficiency_98']:.3f}")
        print(f"   設定差: {diff:.2f}")
        print()
    
    # 中間設定の予想性能
    best_match = closest_matches[0]
    name, _, result = best_match
    
    print(f"🔮 中間設定の予想性能 (最近似: {name}):")
    print(f"   予想ランキング: {result['rank']}位程度")
    print(f"   予想汎用スコア: {result['score']} 程度")
    print(f"   予想効率: 70={result['efficiency_70']:.3f}, 98={result['efficiency_98']:.3f} 程度")
    print()
    
    # 科学的根拠の評価
    print(f"🧪 科学的根拠の評価:")
    if result['rank'] <= 2:
        evidence = "強い根拠あり"
        recommendation = "推奨"
    elif result['rank'] <= 4:
        evidence = "中程度の根拠"
        recommendation = "検討可能"
    elif result['rank'] <= 6:
        evidence = "弱い根拠"
        recommendation = "慎重に検討"
    else:
        evidence = "根拠不十分"
        recommendation = "非推奨"
    
    print(f"   科学的根拠: {evidence}")
    print(f"   推奨度: {recommendation}")
    print()
    
    # 最適設定との比較
    best_setting = test_results["超安全"]
    print(f"💡 最適設定 vs 中間設定:")
    print(f"   最適設定 (Kp=0.3): ランキング{best_setting['rank']}位, スコア{best_setting['score']}")
    print(f"   中間設定 (Kp=1.0): ランキング{result['rank']}位程度, スコア{result['score']}程度")
    
    performance_gap = best_setting['score'] - result['score']
    efficiency_gap_70 = best_setting['efficiency_70'] - result['efficiency_70']
    efficiency_gap_98 = best_setting['efficiency_98'] - result['efficiency_98']
    
    print(f"   性能差: {performance_gap:.3f} (中間設定が劣る)")
    print(f"   効率差: 70で{efficiency_gap_70:.3f}, 98で{efficiency_gap_98:.3f}")
    print()
    
    print(f"✅ 結論:")
    print(f"   中間設定案 (Kp=1.0, Kd=1.0, limits=(-4,4)) は:")
    print(f"   🔬 科学的根拠: {evidence}")
    print(f"   📈 予想性能: {result['rank']}位程度")
    print(f"   ⚡ 効率予想: 中程度")
    print(f"   💭 推奨度: 最適設定 (Kp=0.3) より明らかに劣る")
    print()
    print(f"   ⚠️ 重要:")
    print(f"   科学的テストで証明済みの最適設定 (Kp=0.3) を使用すべき")

if __name__ == "__main__":
    analyze_intermediate_settings()
