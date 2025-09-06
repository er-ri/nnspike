#!/usr/bin/env python3
"""
Speed Command vs Actual Analysis for Smoothing Implementation

HIGH_SPEED_BASE=98での速度指令と実測値の詳細分析
スムージング実装のための基礎データ収集と分析

目的:
1. 速度指令と実測値の乖離パターン分析
2. スムージングが必要な場面の特定
3. スムージングアルゴリズムの設計指針導出
4. PID制御への影響評価
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import argparse
from pathlib import Path

# プロットの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class SpeedCommandAnalyzer:
    def __init__(self, high_speed_base=98):
        self.high_speed_base = high_speed_base
        self.df: pd.DataFrame | None = None
        self.analysis_results = {}
        
    def load_sensor_data(self, data_text=None, csv_file=None):
        """センサーデータの読み込み"""
        if csv_file:
            self.df = pd.read_csv(csv_file)
        elif data_text:
            self.df = pd.read_csv(StringIO(data_text), sep='\t')
        else:
            raise ValueError("data_text or csv_file must be provided")
            
        self._prepare_data()
        print(f"データ読み込み完了: {len(self.df)}行")
        
    def _prepare_data(self):
        """データの前処理と派生変数の計算"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_sensor_data first.")
            
        # 基本的な派生変数
        self.df['time_step'] = range(len(self.df))
        self.df['motor_a_speed_abs'] = abs(self.df['motor_a_speed'])
        self.df['motor_b_speed_abs'] = abs(self.df['motor_b_speed'])
        self.df['motor_a_power_abs'] = abs(self.df['motor_a_power'])
        self.df['motor_b_power_abs'] = abs(self.df['motor_b_power'])
        
        # 速度指令推定（HIGH_SPEED_BASEからの推定）
        self.df['estimated_command_a'] = self.high_speed_base  # 基本的にはBASE値
        self.df['estimated_command_b'] = self.high_speed_base
        
        # 速度指令vs実測の差分
        self.df['speed_error_a'] = self.df['motor_a_speed_abs'] - self.df['estimated_command_a']
        self.df['speed_error_b'] = self.df['motor_b_speed_abs'] - self.df['estimated_command_b']
        
        # 速度変化率（加速度的指標）
        self.df['speed_change_a'] = self.df['motor_a_speed'].diff()
        self.df['speed_change_b'] = self.df['motor_b_speed'].diff()
        self.df['speed_change_abs_a'] = abs(self.df['speed_change_a'])
        self.df['speed_change_abs_b'] = abs(self.df['speed_change_b'])
        
        # 左右モーター差分（ステアリング指標）
        self.df['steering_diff'] = self.df['motor_a_speed'] - self.df['motor_b_speed']
        self.df['steering_diff_abs'] = abs(self.df['steering_diff'])
        
        # スムージング必要度指標
        self.df['smoothing_need_a'] = self.df['speed_change_abs_a'] > 20  # 閾値は調整可能
        self.df['smoothing_need_b'] = self.df['speed_change_abs_b'] > 20
        
    def analyze_command_vs_actual(self):
        """速度指令vs実測の詳細分析"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_sensor_data first.")
            
        print("=== 速度指令vs実測分析 ===")
        
        # 基本統計
        stats = {
            'motor_a_avg_speed': self.df['motor_a_speed_abs'].mean(),
            'motor_b_avg_speed': self.df['motor_b_speed_abs'].mean(),
            'motor_a_speed_std': self.df['motor_a_speed_abs'].std(),
            'motor_b_speed_std': self.df['motor_b_speed_abs'].std(),
            'motor_a_error_mean': self.df['speed_error_a'].mean(),
            'motor_b_error_mean': self.df['speed_error_b'].mean(),
            'motor_a_error_std': self.df['speed_error_a'].std(),
            'motor_b_error_std': self.df['speed_error_b'].std(),
        }
        
        self.analysis_results['basic_stats'] = stats
        
        print(f"Motor A - 平均速度: {stats['motor_a_avg_speed']:.1f}, 標準偏差: {stats['motor_a_speed_std']:.1f}")
        print(f"Motor B - 平均速度: {stats['motor_b_avg_speed']:.1f}, 標準偏差: {stats['motor_b_speed_std']:.1f}")
        print(f"Motor A - 速度誤差平均: {stats['motor_a_error_mean']:.1f}, 標準偏差: {stats['motor_a_error_std']:.1f}")
        print(f"Motor B - 速度誤差平均: {stats['motor_b_error_mean']:.1f}, 標準偏差: {stats['motor_b_error_std']:.1f}")
        
        return stats
        
    def analyze_speed_changes(self):
        """速度変化パターンの分析"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_sensor_data first.")
            
        print("\n=== 速度変化パターン分析 ===")
        
        # 速度変化の統計
        change_stats = {
            'motor_a_change_mean': self.df['speed_change_a'].mean(),
            'motor_b_change_mean': self.df['speed_change_b'].mean(),
            'motor_a_change_std': self.df['speed_change_a'].std(),
            'motor_b_change_std': self.df['speed_change_b'].std(),
            'motor_a_change_abs_mean': self.df['speed_change_abs_a'].mean(),
            'motor_b_change_abs_mean': self.df['speed_change_abs_b'].mean(),
        }
        
        # 大きな変化のあるポイント
        large_change_threshold = 20
        large_changes_a = len(self.df[self.df['speed_change_abs_a'] > large_change_threshold])
        large_changes_b = len(self.df[self.df['speed_change_abs_b'] > large_change_threshold])
        
        print(f"Motor A - 平均速度変化: {change_stats['motor_a_change_mean']:.1f}, 標準偏差: {change_stats['motor_a_change_std']:.1f}")
        print(f"Motor B - 平均速度変化: {change_stats['motor_b_change_mean']:.1f}, 標準偏差: {change_stats['motor_b_change_std']:.1f}")
        print(f"大きな速度変化({large_change_threshold}以上) - Motor A: {large_changes_a}回, Motor B: {large_changes_b}回")
        
        self.analysis_results['change_stats'] = change_stats
        self.analysis_results['large_changes'] = {'a': large_changes_a, 'b': large_changes_b}
        
        return change_stats
        
    def analyze_smoothing_opportunities(self):
        """スムージング機会の分析"""
        print("\n=== スムージング機会分析 ===")
        
        # スムージングが必要な場面の特定
        smoothing_needed_a = self.df['smoothing_need_a'].sum()
        smoothing_needed_b = self.df['smoothing_need_b'].sum()
        total_points = len(self.df)
        
        print(f"スムージング推奨ポイント:")
        print(f"Motor A: {smoothing_needed_a}/{total_points} ({smoothing_needed_a/total_points*100:.1f}%)")
        print(f"Motor B: {smoothing_needed_b}/{total_points} ({smoothing_needed_b/total_points*100:.1f}%)")
        
        # ステアリング制御への影響分析
        steering_stats = {
            'steering_diff_mean': self.df['steering_diff'].mean(),
            'steering_diff_std': self.df['steering_diff'].std(),
            'steering_diff_abs_mean': self.df['steering_diff_abs'].mean(),
        }
        
        print(f"\nステアリング制御分析:")
        print(f"左右差平均: {steering_stats['steering_diff_mean']:.1f}")
        print(f"左右差標準偏差: {steering_stats['steering_diff_std']:.1f}")
        print(f"絶対左右差平均: {steering_stats['steering_diff_abs_mean']:.1f}")
        
        self.analysis_results['smoothing_opportunities'] = {
            'needed_a': smoothing_needed_a,
            'needed_b': smoothing_needed_b,
            'percentage_a': smoothing_needed_a/total_points*100,
            'percentage_b': smoothing_needed_b/total_points*100,
            'steering_stats': steering_stats
        }
        
        return self.analysis_results['smoothing_opportunities']
        
    def plot_comprehensive_analysis(self):
        """包括的な分析グラフ"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # 1. 速度指令vs実測の時系列
        axes[0,0].plot(self.df['time_step'], self.df['motor_a_speed_abs'], 
                      label='Motor A Actual', color='blue', alpha=0.8)
        axes[0,0].plot(self.df['time_step'], self.df['motor_b_speed_abs'], 
                      label='Motor B Actual', color='red', alpha=0.8)
        axes[0,0].axhline(y=self.high_speed_base, color='green', linestyle='--', 
                         label=f'Command ({self.high_speed_base})')
        axes[0,0].set_title('Speed Command vs Actual Over Time')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Speed')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. 速度誤差の時系列
        axes[0,1].plot(self.df['time_step'], self.df['speed_error_a'], 
                      label='Motor A Error', color='blue', alpha=0.7)
        axes[0,1].plot(self.df['time_step'], self.df['speed_error_b'], 
                      label='Motor B Error', color='red', alpha=0.7)
        axes[0,1].axhline(y=0, color='green', linestyle='--', label='Perfect Match')
        axes[0,1].set_title('Speed Error (Actual - Command)')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Speed Error')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. 速度変化率
        axes[1,0].plot(self.df['time_step'], self.df['speed_change_a'], 
                      label='Motor A Change', color='blue', alpha=0.7)
        axes[1,0].plot(self.df['time_step'], self.df['speed_change_b'], 
                      label='Motor B Change', color='red', alpha=0.7)
        axes[1,0].axhline(y=20, color='orange', linestyle='--', label='Smoothing Threshold')
        axes[1,0].axhline(y=-20, color='orange', linestyle='--')
        axes[1,0].set_title('Speed Change Rate')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Speed Change')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 4. 速度変化の絶対値
        axes[1,1].plot(self.df['time_step'], self.df['speed_change_abs_a'], 
                      label='Motor A |Change|', color='blue', alpha=0.7)
        axes[1,1].plot(self.df['time_step'], self.df['speed_change_abs_b'], 
                      label='Motor B |Change|', color='red', alpha=0.7)
        axes[1,1].axhline(y=20, color='orange', linestyle='--', label='Smoothing Threshold')
        axes[1,1].set_title('Absolute Speed Change')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('|Speed Change|')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # 5. ステアリング差分
        axes[2,0].plot(self.df['time_step'], self.df['steering_diff'], 
                      label='L-R Speed Diff', color='purple', alpha=0.7)
        axes[2,0].axhline(y=0, color='green', linestyle='--', label='Straight')
        axes[2,0].set_title('Steering Difference (Motor A - Motor B)')
        axes[2,0].set_xlabel('Time Step')
        axes[2,0].set_ylabel('Speed Difference')
        axes[2,0].legend()
        axes[2,0].grid(True)
        
        # 6. スムージング必要度のヒートマップ
        smoothing_matrix = np.column_stack([
            self.df['smoothing_need_a'].astype(int),
            self.df['smoothing_need_b'].astype(int)
        ])
        im = axes[2,1].imshow(smoothing_matrix.T, aspect='auto', cmap='RdYlGn_r', 
                             extent=[0, len(self.df), 0, 2])
        axes[2,1].set_title('Smoothing Need Heatmap')
        axes[2,1].set_xlabel('Time Step')
        axes[2,1].set_ylabel('Motor (0=A, 1=B)')
        axes[2,1].set_yticks([0.5, 1.5])
        axes[2,1].set_yticklabels(['Motor A', 'Motor B'])
        plt.colorbar(im, ax=axes[2,1], label='Needs Smoothing')
        
        plt.tight_layout()
        plt.show()
        
    def recommend_smoothing_strategy(self):
        """スムージング戦略の推奨"""
        print("\n=== スムージング戦略推奨 ===")
        
        # 分析結果に基づく推奨
        stats = self.analysis_results.get('basic_stats', {})
        changes = self.analysis_results.get('change_stats', {})
        opportunities = self.analysis_results.get('smoothing_opportunities', {})
        
        speed_volatility = max(stats.get('motor_a_speed_std', 0), 
                              stats.get('motor_b_speed_std', 0))
        change_volatility = max(changes.get('motor_a_change_abs_mean', 0),
                               changes.get('motor_b_change_abs_mean', 0))
        
        print("1. スムージング手法推奨:")
        if speed_volatility > 25:
            print("   → 移動平均フィルター（窓サイズ3-5）")
        elif change_volatility > 15:
            print("   → 指数平滑法（α=0.7-0.8）")
        else:
            print("   → 軽微なスムージング（α=0.9）またはスムージング不要")
            
        print("\n2. 実装優先度:")
        max_smoothing_need = max(opportunities.get('percentage_a', 0),
                                opportunities.get('percentage_b', 0))
        if max_smoothing_need > 30:
            print("   → 高優先度: 即座にスムージング実装を推奨")
        elif max_smoothing_need > 15:
            print("   → 中優先度: パフォーマンス向上のためスムージング検討")
        else:
            print("   → 低優先度: 現状でも安定、スムージングは任意")
            
        print("\n3. PID制御への影響考慮:")
        steering_std = opportunities.get('steering_stats', {}).get('steering_diff_std', 0)
        if steering_std > 20:
            print("   → PID設定の見直しが必要（スムージング前に）")
        else:
            print("   → PID制御は安定、スムージング実装可能")
            
        return {
            'recommended_method': 'exponential' if change_volatility > 15 else 'moving_average',
            'priority': 'high' if max_smoothing_need > 30 else 'medium' if max_smoothing_need > 15 else 'low',
            'pid_stable': steering_std <= 20
        }

def main():
    parser = argparse.ArgumentParser(description='Speed Command vs Actual Analysis')
    parser.add_argument('--base-speed', type=int, default=98, help='HIGH_SPEED_BASE value')
    parser.add_argument('--data-file', type=str, help='CSV file with sensor data')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # アナライザーのインスタンス化
    analyzer = SpeedCommandAnalyzer(high_speed_base=args.base_speed)
    
    # データ読み込み（サンプルデータまたはファイル）
    if args.data_file:
        analyzer.load_sensor_data(csv_file=args.data_file)
    else:
        # サンプルデータ（実際のデータに置き換え）
        sample_data = '''motor_a_position	motor_a_relative_position	motor_a_speed	motor_a_power	motor_b_position	motor_b_relative_position	motor_b_speed	motor_b_power
94	0	0	0	27	0	0	0
94	0	0	0	27	0	0	0
93	0	0	0	29	2	2	0'''
        analyzer.load_sensor_data(data_text=sample_data)
    
    # 分析実行
    analyzer.analyze_command_vs_actual()
    analyzer.analyze_speed_changes()
    analyzer.analyze_smoothing_opportunities()
    
    # 可視化
    analyzer.plot_comprehensive_analysis()
    
    # 推奨戦略
    recommendations = analyzer.recommend_smoothing_strategy()
    
    print(f"\n=== 分析完了 ===")
    print(f"HIGH_SPEED_BASE: {args.base_speed}")
    print(f"推奨スムージング手法: {recommendations['recommended_method']}")
    print(f"実装優先度: {recommendations['priority']}")

if __name__ == "__main__":
    main()
