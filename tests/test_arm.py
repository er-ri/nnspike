from nnspike.unit import ETRobot
import time

if __name__ == "__main__":
    et = ETRobot()
    print("Arm up")
    et.move_arm(1, 1.0)  # アームを上げる
    time.sleep(0.5)
    print("Arm down")
    et.move_arm(0, 1.0)  # アームを下げる
    time.sleep(0.5)
    print("Arm stop")
    et.move_arm(2, 0.5)  # アームを止める
