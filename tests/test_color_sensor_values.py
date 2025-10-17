
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from nnspike.unit.action_chain import ActionChain
from nnspike.unit.etrobot import ETRobot

class DummyPID:
    def update(self, theta):
        return 0

class TestGetColorSensorValues(unittest.TestCase):
    def setUp(self):
        self.et = ETRobot()
        self.pid = DummyPID()
        self.chain = ActionChain(self.et, course="right", course_type="upper", pid=self.pid)

    def test_get_color_sensor_values(self):
        # run_realtime.pyのように複数回受信・待機しながらカラー値を取得
        import time
        results = []
        for i in range(20):
            color_value, color_type = self.et.get_color_sensor()
            print(f"[{i+1}] color_value={color_value}, color_type={color_type}")
            results.append((color_value, color_type))
            time.sleep(0.1)
        # すべての取得値を検証
        for color_value, color_type in results:
            self.assertIsInstance(color_value, int)
            self.assertIsInstance(color_type, str)
            self.assertIn(color_type, ["black", "white", "other", "unknown"])

if __name__ == "__main__":
    unittest.main()
