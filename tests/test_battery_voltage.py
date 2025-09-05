#!/usr/bin/env python3
"""
SPIKEハブのバッテリー残量チェックプログラム（本番前確認用）
"""

import sys
import time
from pathlib import Path

# nnspike モジュールのパスを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nnspike.unit.etrobot import ETRobot


def check_battery():
    """SPIKEハブのバッテリー残量をチェック（本番前確認用）"""
    
    print("=" * 50)
    print("SPIKE Hub Battery Check - 本番前バッテリー確認")
    print("=" * 50)
    
    et = ETRobot()
    
    try:
        print("SPIKEハブに接続中...")
        time.sleep(1)  # 初期化待ち
        
        # バッテリー情報を5回測定して平均を取る
        voltage_readings = []
        percent_readings = []
        
        for i in range(5):
            status = et.get_spike_status()
            battery_voltage = getattr(status.battery, 'voltage', None)
            battery_percent = getattr(status.battery, 'percent', None)
            
            if battery_voltage is not None:
                voltage_readings.append(battery_voltage)
                percent_readings.append(battery_percent)
                print(f"測定{i+1}: {battery_voltage:.2f}V, {battery_percent:.1f}%")
            else:
                print(f"測定{i+1}: バッテリー情報取得失敗")
            
            time.sleep(0.5)
        
        print("\n" + "=" * 50)
        
        if voltage_readings:
            avg_voltage = sum(voltage_readings) / len(voltage_readings)
            avg_percent = sum(percent_readings) / len(percent_readings)
            
            print(f"📊 バッテリー状態:")
            print(f"   電圧: {avg_voltage:.2f}V")
            print(f"   残量: {avg_percent:.1f}%")
            
            # バッテリー状態判定と推奨設定
            print(f"\n🔋 バッテリー判定:")
            if avg_voltage >= 8.5:
                status_msg = "🟢 フル充電 - 最高性能"
                recommended_speed = 95
                run_recommendation = "✅ 本番実行OK"
            elif avg_voltage >= 8.0:
                status_msg = "🟡 良好 - 通常性能"
                recommended_speed = 85
                run_recommendation = "✅ 本番実行OK"
            elif avg_voltage >= 7.5:
                status_msg = "🟠 中程度 - 性能低下"
                recommended_speed = 75
                run_recommendation = "⚠️  本番実行注意（充電推奨）"
            else:
                status_msg = "🔴 要充電 - 大幅性能低下"
                recommended_speed = 65
                run_recommendation = "❌ 本番実行非推奨（要充電）"
            
            print(f"   {status_msg}")
            print(f"\n⚙️  推奨HIGH_SPEED_BASE: {recommended_speed}")
            print(f"🏃 本番実行判定: {run_recommendation}")
            
        else:
            print("❌ バッテリー情報を取得できませんでした")
            print("   - SPIKEハブの電源を確認してください")
            print("   - USB接続を確認してください")
        
        print("\n" + "=" * 50)
    
    except KeyboardInterrupt:
        print("\n中断されました")
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("SPIKEハブとの接続を確認してください")
    finally:
        et.stop()
        print("バッテリーチェック完了\n")


if __name__ == "__main__":
    check_battery()

