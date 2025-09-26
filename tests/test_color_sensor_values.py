
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
        # カラーセンサー値を取得
        values = self.chain.get_color_sensor_values()
        print("取得したカラーセンサー値:", values)
        # 値の妥当性チェック
        self.assertIsInstance(values, dict)
        self.assertIn("color", values)
        self.assertIn("color_type", values)
        self.assertIsInstance(values["color"], int)
        self.assertIsInstance(values["color_type"], str)
        # 代表的なtype名
        self.assertIn(values["color_type"], ["black", "white", "other", "unknown"])

if __name__ == "__main__":
    unittest.main()
