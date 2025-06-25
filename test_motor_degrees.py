#!/usr/bin/env python3
"""
左右モーターのset_motor_degrees動作テスト用スクリプト
- 左右それぞれ指定角度だけ回転させ、動作を確認する
- 実行後は自動で停止
"""
import time
from nnspike.unit import ETRobot
TURN_LEFT_DEGREES = -180  # 45度左方向に回転する左モーター角度
TURN_RIGHT_DEGREES = 180  # 45度左方向に回転する右モーター角度
ARC_POWER = 50
ARC_DURATION = 5.0  # 弧を描く時間（仮: 2m相当、要調整）
FORWARD_POWER = 40
FORWARD_DISTANCE = 2.0  # m単位
SPEED_MPS = 0.5  # 仮: 0.5m/s（要実測で調整）
FORWARD_DURATION = FORWARD_DISTANCE / SPEED_MPS

def main():
    et = ETRobot()
    try:
        # 45度左回転テスト
        print("turn_left(angle=45)で45度左回転テスト")
        et.turn_left(angle=45)
        time.sleep(3)
        # 右弧旋回テスト
        print(f"move_right_arc(duration={ARC_DURATION}, power={ARC_POWER})で右弧旋回テスト")
        et.move_right_arc(duration=ARC_DURATION, power=ARC_POWER)
        et.brake()
        time.sleep(3)
    except KeyboardInterrupt:
        print("中断されました。停止します。")
        et.brake()
    finally:
        et.stop()

if __name__ == "__main__":
    main()
