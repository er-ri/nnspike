#!/usr/bin/env python3
"""
SPIKE Hub USB受信信号内容＆周期確認テスト

- Spike HubのUSBポート自動検出
- 受信した生データ（raw_data）と受信周期（ms）をprint
"""
import os
import sys
import time
import platform
import serial
import serial.tools.list_ports

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import importlib.util
import sys
import os

# --- ETRobot受信テスト用最小クラス ---
class SpikeStatus:
    def __init__(self, raw_data):
        self.raw_data = raw_data
    def _parse_data(self):
        import json, time
        try:
            s = self.raw_data.decode(errors='ignore').strip()
            if s.endswith('\r'):
                s = s[:-1]
            json_data = json.loads(s)
            message_type = json_data.get("m", -1)
            payload = json_data.get("p", [])
            result = {
                "message_type": message_type,
                "timestamp": time.time(),
                "raw": json_data,
                "motors": {},
                "sensors": {},
                "battery": {},
            }
            # Motor A (port 48)
            if len(payload) > 0 and isinstance(payload[0], list) and payload[0][0] == 48:
                result["motors"]["A"] = {
                    "speed": (payload[0][1][0] if len(payload[0][1]) > 0 else None),
                    "relative_position": (payload[0][1][1] if len(payload[0][1]) > 2 else None),
                    "position": (payload[0][1][2] if len(payload[0][1]) > 2 else None),
                    "power": (payload[0][1][3] if len(payload[0][1]) > 3 else None),
                }
            # Motor B (port 48)
            if len(payload) > 1 and isinstance(payload[1], list) and payload[1][0] == 48:
                result["motors"]["B"] = {
                    "speed": (payload[1][1][0] if len(payload[1][1]) > 0 else None),
                    "relative_position": (payload[1][1][1] if len(payload[1][1]) > 2 else None),
                    "position": (payload[1][1][2] if len(payload[1][1]) > 2 else None),
                    "power": (payload[1][1][3] if len(payload[1][1]) > 3 else None),
                }
            # Motor C (port 49)
            if len(payload) > 2 and isinstance(payload[2], list) and payload[2][0] == 49:
                result["motors"]["C"] = {
                    "speed": (payload[2][1][0] if len(payload[2][1]) > 0 else None),
                    "relative_position": (payload[2][1][1] if len(payload[2][1]) > 2 else None),
                    "position": (payload[2][1][2] if len(payload[2][1]) > 2 else None),
                    "power": (payload[2][1][3] if len(payload[2][1]) > 3 else None),
                }
            # Force sensor (port 63)
            if len(payload) > 3 and isinstance(payload[3], list) and payload[3][0] == 63:
                result["sensors"]["force"] = payload[3][1][2] if len(payload[3][1]) > 2 else None
            # Color sensor (port 61)
            if len(payload) > 4 and isinstance(payload[4], list) and payload[4][0] == 61:
                color_payload = payload[4][1] if len(payload[4]) > 1 else []
                color_data = [
                    color_payload[0] if len(color_payload) > 0 else None,
                    None,
                    color_payload[2] if len(color_payload) > 2 else None,
                    color_payload[3] if len(color_payload) > 3 else None,
                    color_payload[4] if len(color_payload) > 4 else None,
                ]
                result["sensors"]["color"] = {
                    "reflected": color_data[2],
                    "ambient": color_data[3],
                    "color": color_data[4],
                }
            # Distance sensor (port 62)
            if len(payload) > 5 and isinstance(payload[5], list) and payload[5][0] == 62:
                result["sensors"]["distance"] = payload[5][1][0] if len(payload[5][1]) > 0 else None
            # Accelerometer
            if len(payload) > 6 and isinstance(payload[6], list) and len(payload[6]) == 3:
                result["sensors"]["accelerometer"] = {
                    "x": payload[6][0],
                    "y": payload[6][1],
                    "z": payload[6][2],
                }
            # Gyroscope
            if len(payload) > 7 and isinstance(payload[7], list) and len(payload[7]) == 3:
                result["sensors"]["gyroscope"] = {
                    "x": payload[7][0],
                    "y": payload[7][1],
                    "z": payload[7][2],
                }
            # YawPitchRoll
            if len(payload) > 8 and isinstance(payload[8], list) and len(payload[8]) == 3:
                result["sensors"]["yaw_pitch_roll"] = {
                    "x": payload[8][0],
                    "y": payload[8][1],
                    "z": payload[8][2],
                }
            # Battery status message
            if message_type == 2 and len(payload) > 1:
                result["battery"] = {
                    "voltage": payload[0] if len(payload) > 0 else None,
                    "percent": payload[1] if len(payload) > 1 else None,
                }
            return result
        except Exception as e:
            return f"ParseError: {e}"

class ETRobot:
    def __init__(self, port):
        self.ser = serial.Serial(port, baudrate=115200, timeout=0.1)
    def get_spike_status(self):
        received_data = b""
        while True:
            chunk = self.ser.read_until(expected=b"\r")
            received_data += chunk
            if received_data.strip().endswith(b'}'):
                break
        return SpikeStatus(received_data)
    def stop(self):
        self.ser.close()

def find_spike_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if any(keyword in port.description.lower() for keyword in ['spike', 'lego', 'usb serial', 'usb-serial']):
            return port.device
    # Windows用: COM3/COM4/COM5を順に試す
    if platform.system() == "Windows":
        for com_port in ["COM3", "COM4", "COM5"]:
            try:
                # ポートが存在するか確認
                if any(port.device == com_port for port in ports):
                    return com_port
                # 存在しない場合でもopenできるか試す
                s = serial.Serial(com_port)
                s.close()
                return com_port
            except Exception:
                continue
        return None
    # Linux用デフォルト
    return None

def main():
    print("SPIKE Hub USB受信信号内容＆周期確認テスト")
    port = find_spike_port()
    print(f"使用ポート: {port}")
    if port is None:
        print("SPIKE HubのCOMポートが見つかりません。USB接続を確認してください。")
        return
    et = ETRobot(port=port)
    print("ETRobot初期化完了。受信開始...")
    duration = 2.0  # 秒
    start = time.time()
    m_prev_time = {}  # m値ごとの前回受信時刻
    m_cycle = {}      # m値ごとの周期リスト
    count = 0
    while time.time() - start < duration:
        status = et.get_spike_status()
        now_time = time.time()
        # m値抽出
        try:
            raw_str = status.raw_data.decode(errors='ignore')
            import re
            m_match = re.search(r'"m"\s*:\s*(\d+)', raw_str)
            m_val = int(m_match.group(1)) if m_match else None
        except Exception:
            m_val = None
        cycle = None
        if m_val is not None:
            if m_val in m_prev_time:
                cycle = (now_time - m_prev_time[m_val]) * 1000
                if m_val not in m_cycle:
                    m_cycle[m_val] = []
                m_cycle[m_val].append(cycle)
            m_prev_time[m_val] = now_time
        # パース結果も表示
        parsed = status._parse_data()
        cycle_str = f"{cycle:.2f}" if cycle is not None else "None"
        if isinstance(parsed, dict):
            motors = parsed.get('motors', {})
            sensors = parsed.get('sensors', {})
            battery = parsed.get('battery', {})
            volt = battery.get('voltage')
            perc = battery.get('percent')
            if m_val == 2:
                # バッテリー情報のみ表示（小数点以下2桁まで）
                volt_str = f"{volt:.5f}" if volt is not None else "None"
                perc_str = f"{perc:.5f}" if perc is not None else "None"
                print(f"[{count}] m={m_val} cycle={cycle_str}ms batt={volt_str}V/{perc_str}%")
            else:
                # 通常のセンサ・モーター情報表示
                a = motors.get('A', {})
                b = motors.get('B', {})
                c = motors.get('C', {})
                a_str = f"A[speed={a.get('speed')},relpos={a.get('relative_position')},pos={a.get('position')},pwr={a.get('power')}]" if a else "A[-]"
                b_str = f"B[speed={b.get('speed')},relpos={b.get('relative_position')},pos={b.get('position')},pwr={b.get('power')}]" if b else "B[-]"
                c_str = f"C[speed={c.get('speed')},relpos={c.get('relative_position')},pos={c.get('position')},pwr={c.get('power')}]" if c else "C[-]"
                dist = sensors.get('distance')
                force = sensors.get('force')
                color = sensors.get('color', {})
                color_str = f"refl={color.get('reflected')} amb={color.get('ambient')} color={color.get('color')}" if color else "color[-]"
                acc = sensors.get('accelerometer', {})
                gyro = sensors.get('gyroscope', {})
                ypr = sensors.get('yaw_pitch_roll', {})
                acc_str = f"acc=({acc.get('x')},{acc.get('y')},{acc.get('z')})" if acc else "acc[-]"
                gyro_str = f"gyro=({gyro.get('x')},{gyro.get('y')},{gyro.get('z')})" if gyro else "gyro[-]"
                ypr_str = f"ypr=({ypr.get('x')},{ypr.get('y')},{ypr.get('z')})" if ypr else "ypr[-]"
                print(f"[{count}] m={m_val} cycle={cycle_str}ms {a_str} {b_str} {c_str} dist={dist} force={force} {color_str} {acc_str} {gyro_str} {ypr_str} batt={volt}V/{perc}%")
        else:
            print(f"[{count}] m={m_val} cycle={cycle_str}ms parse_error: {parsed}")
        count += 1
        time.sleep(0.01)
    # mごとの周期統計表示
    print("\n--- mごとの周期統計 ---")
    for m_val, cycles in m_cycle.items():
        avg = sum(cycles)/len(cycles) if cycles else 0
        print(f"m={m_val}: count={len(cycles)}, avg={avg:.2f}ms")
    et.stop()
    print("受信テスト終了")

if __name__ == "__main__":
    main()
