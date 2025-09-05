#!/usr/bin/env python3
"""
SPIKE Battery Voltage Test Program

このプログラムはSPIKEハブのバッテリー電圧を測定します。
バッテリー情報がコメントアウトされていても、raw dataから直接取得します。
"""

import sys
import time
import json
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nnspike.unit import ETRobot


def test_battery_voltage():
    """SPIKEハブのバッテリー電圧を測定する"""
    
    print("SPIKE Battery Voltage Test")
    print("=" * 40)
    
    # ETRobotインスタンスを作成
    et = ETRobot()
    
    try:
        print("SPIKEハブに接続中...")
        
        # 複数回測定して平均を取る
        voltage_readings = []
        percent_readings = []
        
        for i in range(10):
            status = et.get_spike_status()
            
            # Method 1: 通常のバッテリー情報取得
            battery_voltage = getattr(status.battery, 'voltage', None)
            battery_percent = getattr(status.battery, 'percent', None)
            
            # Method 2: raw dataから直接取得
            raw_data = getattr(status, 'raw_data', {})
            message_type = raw_data.get('m', -1)
            payload = raw_data.get('p', [])
            
            raw_voltage = None
            raw_percent = None
            
            if message_type == 2 and len(payload) >= 2:
                raw_voltage = payload[0]
                raw_percent = payload[1]
            
            print(f"測定 {i+1:2d}: ", end="")
            
            if battery_voltage is not None:
                print(f"電圧={battery_voltage:.2f}V, 残量={battery_percent:.1f}%", end="")
                voltage_readings.append(battery_voltage)
                percent_readings.append(battery_percent)
            elif raw_voltage is not None:
                print(f"電圧={raw_voltage:.2f}V, 残量={raw_percent:.1f}% (raw)", end="")
                voltage_readings.append(raw_voltage)
                percent_readings.append(raw_percent)
            else:
                print("バッテリー情報なし", end="")
            
            print(f" (メッセージタイプ: {message_type})")
            
            time.sleep(0.5)
        
        # 結果の集計
        print("\n" + "=" * 40)
        print("測定結果:")
        
        if voltage_readings:
            avg_voltage = sum(voltage_readings) / len(voltage_readings)
            avg_percent = sum(percent_readings) / len(percent_readings)
            min_voltage = min(voltage_readings)
            max_voltage = max(voltage_readings)
            
            print(f"平均電圧: {avg_voltage:.2f}V")
            print(f"平均残量: {avg_percent:.1f}%")
            print(f"電圧範囲: {min_voltage:.2f}V - {max_voltage:.2f}V")
            
            # 動的HIGH_SPEED_BASE推奨値
            print("\n推奨HIGH_SPEED_BASE設定:")
            if avg_voltage > 8.5:
                recommended_speed = 95
                status_msg = "フル充電状態"
            elif avg_voltage > 8.0:
                recommended_speed = 85
                status_msg = "良好"
            elif avg_voltage > 7.5:
                recommended_speed = 75
                status_msg = "中程度"
            else:
                recommended_speed = 65
                status_msg = "要充電"
            
            print(f"HIGH_SPEED_BASE = {recommended_speed} ({status_msg})")
            
        else:
            print("バッテリー情報を取得できませんでした")
            print("以下を確認してください:")
            print("1. SPIKEハブの電源が入っているか")
            print("2. USB接続が正常か")
            print("3. spike_status.pyのバッテリー処理がコメントアウトされていないか")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("SPIKEハブとの接続を確認してください")
    
    finally:
        et.stop()
        print("\nテスト完了")


if __name__ == "__main__":
    test_battery_voltage()
