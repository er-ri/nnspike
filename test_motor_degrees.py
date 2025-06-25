#!/usr/bin/env python3
"""
左右モーターのset_motor_degrees動作テスト用スクリプト
- 左右それぞれ指定角度だけ回転させ、動作を確認する
- 実行後は自動で停止
"""
import time
from nnspike.unit import ETRobot

def main():
    et = ETRobot()
    try:
        print("左右モーター: power=100 で1秒間動作")
        et.set_motor_forward_power(left_power=50, right_power=25)
        time.sleep(1)
        et.brake()
        time.sleep(3)
        print("左モーター: 180度、右モーター: 180度 正回転")
        et.set_motor_degrees(left_degrees=180, right_degrees=180)
        time.sleep(3)
    except KeyboardInterrupt:
        print("中断されました。停止します。")
        et.brake()
    finally:
        et.stop()

if __name__ == "__main__":
    main()
