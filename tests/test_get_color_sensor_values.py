import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from nnspike.unit.action_chain import ActionChain
from nnspike.unit.etrobot import ETRobot

class TestGetColorSensorValues(unittest.TestCase):
    def get_valid_status(self, max_retry=30):
        serial_port = getattr(self.et, '_ETRobot__serial_port', None)
        for _ in range(max_retry):
            raw = serial_port.read_until(expected=b"\r")
            try:
                if raw.startswith(b"{"):
                    data = json.loads(raw.decode("utf-8").strip())
                    if data.get("m") == 0:
                        return raw
            except Exception:
                continue
        return None
    def setUp(self):
        self.et = ETRobot()
        self.chain = ActionChain(self.et, course="right", course_type="upper")

    def test_get_color_sensor_values(self):
        # 正しいJSONが来るまでリトライ
        valid_raw = self.get_valid_status()
        print("valid_raw:", valid_raw)
        # valid_rawのpayloadから[61, ...]の内容をprint
        if valid_raw:
            try:
                data = json.loads(valid_raw.decode("utf-8").strip())
                payload = data.get("p", [])
                color_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 61]
                if color_entries:
                    print("color_entries:", color_entries)
                    print("color_entries[0][1] (len):", color_entries[0][1], len(color_entries[0][1]))
            except Exception as e:
                print("[payload解析エラー]", e)
            self.et.spike_status.update(valid_raw)
        status = self.et.get_spike_status()
        print("status.raw_data:", getattr(status, 'raw_data', None))
        print("status.sensors:", getattr(status, 'sensors', None))
        if hasattr(status, 'sensors') and status.sensors is not None:
            print("status.sensors.color:", getattr(status.sensors, 'color', None))
        values = self.chain.get_color_sensor_values(status)
        print("SPIKEから取得したカラーセンサー値:", values)
        self.assertIn("reflected", values)
        self.assertIn("ambient", values)
        self.assertIn("color", values)
        self.assertIn("is_black", values)

if __name__ == "__main__":
    unittest.main()
