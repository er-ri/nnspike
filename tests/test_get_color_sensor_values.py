import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from nnspike.unit.action_chain import ActionChain
from nnspike.unit.etrobot import ETRobot

class TestGetColorSensorValues(unittest.TestCase):
    def get_valid_status(self, max_retry=20):
        serial_port = getattr(self.et, '_ETRobot__serial_port', None)
        for _ in range(max_retry):
            raw = serial_port.read_until(expected=b"\r")
            try:
                if raw.startswith(b"{"):
                    json.loads(raw.decode("utf-8").strip())
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
        # ここでstatus更新
        if valid_raw:
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
