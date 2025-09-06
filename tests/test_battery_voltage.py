#!/usr/bin/env python3
"""
SPIKEハブのバッテリー残量チェックプログム（本番前確認用）
test内完結型 - utilsに依存しない独立実装
"""

import sys
import time
import json
import serial
from pathlib import Path

# nnspike モジュールのパスを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BatteryChecker:
    """バッテリー情報取得専用クラス（test内完結）"""
    
    def __init__(self, port="/dev/ttyACM0", debug=False):
        self.debug = debug
        self.serial_port = serial.Serial(port=port, baudrate=115200, timeout=2)
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        
    def __parse_battery_message(self, data):
        """バッテリー情報のみを抽出する軽量パーサー"""
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8").strip()
            
            # Clean the data: remove any null bytes or invalid characters  
            data = data.replace('\x00', '').strip()
            
            if self.debug:
                print(f"Parsing data: {data[:100]}")
            
            # Try to find JSON-like content
            if '{' in data and '}' in data:
                start = data.find('{')
                end = data.rfind('}') + 1
                json_str = data[start:end]
                
                parsed = json.loads(json_str)
                
                message_type = parsed.get("m", -1)
                payload = parsed.get("p", [])
                
                if self.debug:
                    print(f"Message type: {message_type}, Payload: {payload}")
                
                # message_type == 2はバッテリー情報
                if message_type == 2 and isinstance(payload, list) and len(payload) > 1:
                    return {
                        "voltage": payload[0] if len(payload) > 0 else None,
                        "percent": payload[1] if len(payload) > 1 else None,
                    }
                    
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if self.debug:
                print(f"Parse error: {e}")
            pass
            
        return None
    
    def get_battery_info(self, max_attempts=25):  # 試行回数を増加
        """
        バッテリー情報を取得
        
        Args:
            max_attempts (int): 最大試行回数
            
        Returns:
            dict: {"voltage": float, "percent": int} または None
        """
        for attempt in range(max_attempts):
            try:
                # シリアルポートからデータを読み取り
                received_data = self.serial_port.read_until(expected=b"\r")
                
                if self.debug and attempt < 3:  # 最初の3回だけデバッグ表示
                    print(f"Attempt {attempt + 1}: Received {len(received_data)} bytes")
                
                if received_data and len(received_data) > 10:  # 最小データサイズチェック
                    battery_info = self.__parse_battery_message(received_data)
                    if battery_info:
                        if self.debug:
                            print(f"✅ Battery found on attempt {attempt + 1}")
                        return battery_info
                        
                # バッテリー情報が見つからない場合は短時間待機
                time.sleep(0.05)  # より短い待機でより多く試行
                
            except Exception as e:
                if self.debug:
                    print(f"Battery read attempt {attempt + 1} failed: {e}")
                time.sleep(0.1)
                
        return None
    
    def close(self):
        """シリアルポートを閉じる"""
        self.serial_port.close()


def check_battery():
    """SPIKEハブのバッテリー残量をチェック（本番前確認用）"""
    
    print("=" * 50)
    print("SPIKE Hub Battery Check - 本番前バッテリー確認")
    print("=" * 50)
    
    battery_checker = None
    
    try:
        print("SPIKEハブに接続中...")
        battery_checker = BatteryChecker(debug=False)  # デバッグ無効化
        time.sleep(1)  # 初期化待ち
        
        # バッテリー情報を5回測定して平均を取る
        voltage_readings = []
        percent_readings = []
        
        print("バッテリー情報収集中...", end="", flush=True)
        
        for i in range(5):
            battery_info = battery_checker.get_battery_info()  # test内完結型バッテリー取得
            
            if battery_info and battery_info.get('voltage') is not None:
                voltage_readings.append(battery_info['voltage'])
                percent_readings.append(battery_info['percent'])
                print(f"\n測定{i+1}: {battery_info['voltage']:.2f}V, {battery_info['percent']:.1f}%")
            else:
                print(f"\n測定{i+1}: バッテリー情報取得失敗")
            
            print(".", end="", flush=True)  # 進行状況表示
            time.sleep(0.3)  # 少し短縮
        
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
        if battery_checker:
            battery_checker.close()
        print("バッテリーチェック完了\n")


if __name__ == "__main__":
    check_battery()

