
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
        color_value, color_type = 0, "unknown"
        for i in range(10):
            color_value, color_type = self.et.get_color_sensor()
            print(f"[{i+1}] color_value={color_value}, color_type={color_type}")
            if color_value != 0:
                break
            time.sleep(0.1)
        # 値の妥当性チェック
        self.assertIsInstance(color_value, int)
        self.assertIsInstance(color_type, str)
        self.assertIn(color_type, ["black", "white", "other", "unknown"])

if __name__ == "__main__":
    unittest.main()
