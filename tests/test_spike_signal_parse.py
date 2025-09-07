#!/usr/bin/env python3
"""
SPIKEハブから受信する全信号をPC直結で解析・表示する独立テストプログラム
- etrobot.pyやspike_status.py等に依存せず、シリアル受信データを直接パース
- 受信した全てのJSONメッセージを生データ・パース結果ともに表示
- Windows/Ubuntu/Raspberry Pi対応（ポート自動検出あり）
"""
import sys
import time
import json
import serial
import platform
from pathlib import Path

# --- SPIKEハブのシリアルポート自動検出 ---
def find_spike_port():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if any(keyword in port.description.lower() for keyword in ['spike', 'lego', 'usb serial', 'usb-serial']):
            print(f"✅ Spikeハブ候補: {port.device} ({port.description})")
            return port.device
    # デフォルトポート
    if platform.system() == "Linux":
        return "/dev/ttyACM0"
    else:
        return "COM3"

# --- SPIKE信号パーサ（JSONメッセージ抽出） ---
def parse_spike_message(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="ignore").strip()
    data = data.replace('\x00', '').strip()
    if '{' in data and '}' in data:
        start = data.find('{')
        end = data.rfind('}') + 1
        json_str = data[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[!] JSONDecodeError: {e}\n  生データ: {repr(data[:100])}")
            return None
    return None

# --- メイン処理 ---
def main():
    port = find_spike_port()
    print(f"\n[INFO] SPIKEハブのシリアルポート: {port}")
    ser = serial.Serial(port=port, baudrate=115200, timeout=2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("\n[INFO] 受信開始。Ctrl+Cで終了\n")
    prev_time = None
    angle = 0.0
    try:
        while True:
            raw = ser.read_until(expected=b"\r")
            now = time.time()
            if not raw or len(raw) < 5:
                continue
            msg = parse_spike_message(raw)
            # サイクル計算
            cycle_ms = None
            dt = None
            if prev_time is not None:
                dt = now - prev_time
                cycle_ms = dt * 1000
            prev_time = now
            # --- 1行横並び表示 ---
            line = ""
            if msg is not None:
                mtype = msg.get("m", None)
                p = msg.get("p", [])
                gyro = None
                if mtype == 0:
                    if len(p) > 7 and isinstance(p[7], list) and len(p[7]) >= 3:
                        gyro = p[7]
                    # 積算角度計算
                    if gyro and dt is not None:
                        angle += gyro[2] * dt  # z軸角速度[deg/s] × 秒 = 角度[deg]
                    if gyro:
                        line += f"[GYRO] x={gyro[0]} y={gyro[1]} z={gyro[2]} | "
                    if gyro and dt is not None:
                        line += f"[ANGLE] {angle:.1f} deg | "
                # mtype==2（バッテリー情報）は何も表示しない
                # サイクル
                if cycle_ms is not None:
                    line += f"[CYCLE] {cycle_ms:.1f} ms"
                if line:
                    print(line.strip())
                else:
                    print("[OK] パース結果: (詳細は省略)")
            else:
                print("[NG] パース失敗 or JSONでない信号")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] 終了します")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
