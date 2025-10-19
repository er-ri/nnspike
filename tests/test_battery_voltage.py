#!/usr/bin/env python3
"""
SPIKEハブのバッテリー残量チェックプログム（本番前確認用）
test内完結型 - utilsに依存しない独立実装
"""

print("ファイル読込テスト")

import sys
import time
import json
import serial
from pathlib import Path

# nnspike モジュールのパスを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import serial.tools.list_ports

def find_spike_port():
    """SPIKE HubのCOMポートを自動検出（Windows対応）"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if any(keyword in port.description.lower() for keyword in ['spike', 'lego', 'usb serial', 'usb-serial']):
            return port.device
    # Windows用: COM3/COM4/COM5を順に試す
    import platform
    if platform.system() == "Windows":
        for com_port in ["COM3", "COM4", "COM5"]:
            if any(port.device == com_port for port in ports):
                return com_port
            s = None
            s = serial.Serial(com_port)
            s.close()
            return com_port
        return None
    return None

class BatteryChecker:
    """バッテリー情報取得専用クラス（test内完結）"""
    
    def __init__(self, port=None, debug=False):
        self.debug = debug
        if port is None:
            port = find_spike_port()
            if port is None:
                raise RuntimeError("SPIKE HubのCOMポートが見つかりません。USB接続を確認してください。")
        self.serial_port = serial.Serial(port=port, baudrate=115200, timeout=2)
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        
    def __parse_battery_message(self, data):
        """m=2のバッテリー情報のみ抽出"""
        if isinstance(data, bytes):
            data = data.decode("utf-8").strip()
        data = data.replace('\x00', '').strip()
        if '{' in data and '}' in data:
            start = data.find('{')
            end = data.rfind('}') + 1
            json_str = data[start:end]
            parsed = json.loads(json_str)
            message_type = parsed.get("m", -1)
            payload = parsed.get("p", [])
            if message_type == 2 and isinstance(payload, list) and len(payload) > 1:
                return {
                    "voltage": payload[0] if len(payload) > 0 else None,
                    "percent": payload[1] if len(payload) > 1 else None,
                }
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
            received_data = self.serial_port.read_until(expected=b"\r")
            # 受信データをデコード
            decoded = received_data.decode("utf-8", errors="ignore").strip()
            # JSONパースしてm値を抽出
            m_val = None
            payload = None
            if '{' in decoded and '}' in decoded:
                parsed = json.loads(decoded[decoded.find('{'):decoded.rfind('}')+1])
                m_val = parsed.get("m", None)
                payload = parsed.get("p", None)
            # m=2のときのみprint
            if m_val == 2:
                print(f"[DEBUG] m=2受信: {decoded}")
                print(f"[DEBUG] payload: {payload}")
                if isinstance(payload, list) and len(payload) > 1:
                    voltage = payload[0]
                    percent = payload[1]
                    print(f"[DEBUG] voltage: {float(voltage):.3f}")
                    print(f"[DEBUG] percent: {float(percent):.3f}")
            if received_data and len(received_data) > 10:
                battery_info = self.__parse_battery_message(received_data)
                if battery_info:
                    if self.debug:
                        print(f"✅ Battery found on attempt {attempt + 1}")
                    return battery_info
            time.sleep(0.05)
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
    print("SPIKEハブに接続中...")
    battery_checker = BatteryChecker(port=None, debug=False)  # ポート自動判定＆デバッグ無効化
    time.sleep(1)  # 初期化待ち

    # バッテリー情報を5回測定して平均を取る
    voltage_readings = []
    percent_readings = []

    print("バッテリー情報収集中...", end="", flush=True)

    for i in range(5):
        battery_info = battery_checker.get_battery_info()  # test内完結型バッテリー取得
        if battery_info and isinstance(battery_info.get('voltage'), (int, float)):
            voltage_readings.append(battery_info['voltage'])
            percent_readings.append(battery_info['percent'] if isinstance(battery_info.get('percent'), (int, float)) else 0)
            v_str = f"{battery_info['voltage']:.5f}" if battery_info['voltage'] is not None else "N/A"
            p_str = f"{battery_info['percent']:.5f}" if battery_info['percent'] is not None else "N/A"
            print(f"\n測定{i+1}: {v_str}V, {p_str}%")
        else:
            print(f"\n測定{i+1}: バッテリー情報取得失敗")
        print(".", end="", flush=True)  # 進行状況表示
        time.sleep(0.3)  # 少し短縮

    print("\n" + "=" * 50)

    if voltage_readings:
        avg_voltage = sum(voltage_readings) / len(voltage_readings)
        avg_percent = sum(percent_readings) / len(percent_readings)
        print(f"📊 バッテリー状態:")
        print(f"   測定値一覧: {[f'{v:.5f}' for v in voltage_readings]}")
        print(f"   電圧: {avg_voltage:.5f}V")
        print(f"   残量: {avg_percent:.5f}%")
        # バッテリー状態判定のみ（スピード推奨は削除）
        print(f"\n🔋 バッテリー判定:")
        if avg_voltage >= 8.500:
            status_msg = f"🟢 フル充電 - 最高性能 (電圧: {avg_voltage:.5f}V)"
            run_recommendation = "✅ 本番実行OK"
        elif avg_voltage >= 8.000:
            status_msg = f"🟡 良好 - 通常性能 (電圧: {avg_voltage:.5f}V)"
            run_recommendation = "✅ 本番実行OK"
        elif avg_voltage >= 7.500:
            status_msg = f"🟠 中程度 - 性能低下 (電圧: {avg_voltage:.5f}V)"
            run_recommendation = "⚠️  本番実行注意（充電推奨）"
        else:
            status_msg = f"🔴 要充電 - 大幅性能低下 (電圧: {avg_voltage:.5f}V)"
            run_recommendation = "❌ 本番実行非推奨（要充電）"
        print(f"   {status_msg}")
        print(f"🏃 本番実行判定: {run_recommendation}")
    else:
        print("❌ バッテリー情報を取得できませんでした")
        print("   - SPIKEハブの電源を確認してください")
        print("   - USB接続を確認してください")
    print("\n" + "=" * 50)

    if battery_checker:
        battery_checker.close()
    print("バッテリーチェック完了\n")


# 直接実行時にバッテリーチェックを起動
if __name__ == "__main__":
    check_battery()

