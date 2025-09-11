import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from nnspike.unit.action_chain import ActionChain

class DummyStatus:
    def __init__(self, color_left=10, color_right=20, color_center=30):
        class Sensors:
            pass
        self.sensors = Sensors()
        self.sensors.color_left = color_left
        self.sensors.color_right = color_right
        self.sensors.color_center = color_center

class DummyETRobot:
    def get_spike_status(self):
        return DummyStatus()

class TestGetColorSensorValues(unittest.TestCase):
    def setUp(self):
        self.et = DummyETRobot()
        self.chain = ActionChain(self.et, course="right")

    def test_get_color_sensor_values(self):
        status = self.et.get_spike_status()
        values = self.chain.get_color_sensor_values(status)
        self.assertIsInstance(values, (list, tuple))
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], 10)
        self.assertEqual(values[1], 20)
        self.assertEqual(values[2], 30)

if __name__ == "__main__":
    unittest.main()
