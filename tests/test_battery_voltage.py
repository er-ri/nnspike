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
        
        # まず基本的な接続確認
        print("接続状態チェック...")
        for i in range(5):
            status = et.get_spike_status()
            message_type = getattr(status, 'message_type', -1)
            print(f"接続テスト {i+1}: メッセージタイプ = {message_type}")
            
            if message_type != -1:
                print("✓ SPIKEハブとの通信確立")
                break
            time.sleep(1)
        else:
            print("✗ SPIKEハブとの通信が確立できません")
            print("SPIKEハブのプログラムが動作しているか確認してください")
            return
        
        print("\nバッテリー情報取得中...")
        
        # 複数回測定して平均を取る
        voltage_readings = []
        percent_readings = []
        
        for i in range(15):  # 回数を増やしてバッテリー情報待ち
            status = et.get_spike_status()
            
            # raw dataの詳細チェック
            raw_data = getattr(status, 'raw_data', {})
            message_type = raw_data.get('m', -1)
            payload = raw_data.get('p', [])
            
            print(f"測定 {i+1:2d}: MSG={message_type}, payload_len={len(payload)}")
            
            # すべてのメッセージタイプの詳細を表示
            if message_type == 0:
                print(f"  → センサーデータ: payload={payload}")
            elif message_type == 2:
                print(f"  → バッテリーデータ: payload={payload}")
                if len(payload) >= 2:
                    voltage = payload[0]
                    percent = payload[1]
                    print(f"  → 電圧={voltage:.2f}V, 残量={percent:.1f}%")
                    voltage_readings.append(voltage)
                    percent_readings.append(percent)
            else:
                print(f"  → 未知のメッセージ: payload={payload}")
            
            # バッテリー情報の状態も確認
            battery_voltage = getattr(status.battery, 'voltage', None)
            battery_percent = getattr(status.battery, 'percent', None)
            print(f"  → status.battery: voltage={battery_voltage}, percent={battery_percent}")
            
            time.sleep(1)
        
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
