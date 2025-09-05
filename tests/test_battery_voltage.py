#!/usr/bin/env python3
"""
SPIKE Battery Voltage Test Program

このプログラムはSPIKEハブのバッテリー電圧を測定します。
バッテリー情報がコメントアウトされていても、raw dataから直接取得します。
"""

import sys
import time
import json
import os

# test_force_sensor.pyと同じパス設定
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nnspike.unit import ETRobot


def test_battery_voltage():
    """SPIKEハブのバッテリー電圧を測定する（test_force_sensor.pyベース）"""
    
    print("SPIKE Battery Voltage Test")
    print("=" * 40)
    
    # test_force_sensor.pyと同じETRobot作成方法
    et = ETRobot()
    
    try:
        # test_force_sensor.pyと同じ初期化チェック
        print("SPIKEハブ初期化チェック...")
        status_init = et.get_spike_status()
        force_val_init = getattr(status_init.sensors, "force", None)
        battery_voltage_init = getattr(status_init.battery, 'voltage', None)
        
        if force_val_init is None:
            time.sleep(1)
            status_init = et.get_spike_status()
            force_val_init = getattr(status_init.sensors, "force", None)
            battery_voltage_init = getattr(status_init.battery, 'voltage', None)
        
        if force_val_init is not None:
            print("✓ フォースセンサー稼働中")
        else:
            print("✗ フォースセンサー未検出")
        
        if battery_voltage_init is not None:
            print("✓ バッテリー情報取得成功")
        else:
            print("✗ バッテリー情報未取得")
        
        print("\nバッテリー情報監視開始...")
        print("フォースセンサーを押すとバッテリー情報が更新される可能性があります")
        print("Ctrl+Cで終了")
        
        voltage_readings = []
        percent_readings = []
        message_counts = {}
        
        count = 0
        while count < 50:  # 50回測定
            count += 1
            status = et.get_spike_status()
            
            # フォースセンサー値
            force_val = getattr(status.sensors, "force", None)
            
            # バッテリー情報
            battery_voltage = getattr(status.battery, 'voltage', None)
            battery_percent = getattr(status.battery, 'percent', None)
            
            # raw data解析
            raw_data = getattr(status, 'raw_data', {})
            message_type = raw_data.get('m', -1)
            
            message_counts[message_type] = message_counts.get(message_type, 0) + 1
            
            print(f"[{count:2d}] MSG={message_type}, フォース={force_val}, ", end="")
            
            if battery_voltage is not None:
                print(f"★電圧={battery_voltage:.2f}V, 残量={battery_percent:.1f}%★")
                voltage_readings.append(battery_voltage)
                percent_readings.append(battery_percent)
            else:
                print("バッテリー情報なし")
            
            time.sleep(0.2)  # test_force_sensor.pyより少し長めに
        
        # 結果表示
        print("\n" + "=" * 40)
        print("測定完了")
        print(f"メッセージタイプ統計: {message_counts}")
        
        if voltage_readings:
            avg_voltage = sum(voltage_readings) / len(voltage_readings)
            avg_percent = sum(percent_readings) / len(percent_readings)
            print(f"バッテリー情報取得成功: {len(voltage_readings)}回")
            print(f"平均電圧: {avg_voltage:.2f}V")
            print(f"平均残量: {avg_percent:.1f}%")
        else:
            print("バッテリー情報を取得できませんでした")
    
    except KeyboardInterrupt:
        print("\nテスト中断")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        et.stop()
        print("テスト完了")


if __name__ == "__main__":
    test_battery_voltage()
