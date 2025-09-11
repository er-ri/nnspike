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
        status = self.et.get_spike_status()
        values = self.chain.get_color_sensor_values(status)
        print("SPIKEから取得したカラーセンサー値:", values)
        self.assertIn("reflected", values)
        self.assertIn("ambient", values)
        self.assertIn("color", values)
        self.assertIn("is_black", values)

if __name__ == "__main__":
    unittest.main()
