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
        # set_motor_degreesテスト: 左右個別・同時
        print("左モーターを180度正転 (右停止)")
        et.set_motor_degrees(180, 0)
        time.sleep(3)
        print("右モーターを180度正転 (左停止)")
        et.set_motor_degrees(0, 180)
        time.sleep(3)
        print("左右同時に-90度逆転")
        et.set_motor_degrees(-90, -90)
        time.sleep(3)
        et.brake()
        time.sleep(1)
    except KeyboardInterrupt:
        print("中断されました。停止します。")
        et.brake()
    finally:
        et.stop()

if __name__ == "__main__":
    main()
