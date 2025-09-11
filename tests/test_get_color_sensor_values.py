import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from nnspike.unit.action_chain import ActionChain
from nnspike.unit.etrobot import ETRobot

class TestGetColorSensorValues(unittest.TestCase):
    def setUp(self):
        self.et = ETRobot()
        self.chain = ActionChain(self.et, course="right", course_type="upper")

    def test_get_color_sensor_values(self):
        # 受信データを直接取得してprint（デバッグ用）
        try:
            serial_port = getattr(self.et, '_ETRobot__serial_port', None)
            if serial_port:
                print('--- 直近のSPIKE受信データ（10件） ---')
                for _ in range(10):
                    raw = serial_port.read_until(expected=b"\r")
                    print("raw_serial:", raw)
        except Exception as e:
            print("[デバッグ用シリアル受信エラー]", e)

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
