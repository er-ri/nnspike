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
        print("左モーター: 720度、右モーター: 720度 正回転")
        et.set_motor_degrees(left_degrees=720, right_degrees=720)
        time.sleep(3)
        print("左モーター: -720度、右モーター: -720度 逆回転")
        et.set_motor_degrees(left_degrees=-720, right_degrees=-720)
        time.sleep(3)
        print("左モーター: 0度、右モーター: 1440度 (右のみ2回転)")
        et.set_motor_degrees(left_degrees=0, right_degrees=1440)
        time.sleep(3)
        print("左モーター: 1440度、右モーター: 0度 (左のみ2回転)")
        et.set_motor_degrees(left_degrees=1440, right_degrees=0)
        time.sleep(3)
        print("テスト完了。停止します。")
        et.brake()
    except KeyboardInterrupt:
        print("中断されました。停止します。")
        et.brake()
    finally:
        et.stop()

if __name__ == "__main__":
    main()
