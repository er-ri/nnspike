#!/usr/bin/env python3
"""
ActionManagerのstep()相当の回避動作（45度左回転→右弧旋回→終了）だけをテストするスクリプト
"""
import time
from nnspike.unit import ETRobot

ARC_POWER = 50
ARC_DURATION = 5.0  # 弧を描く時間（5秒に設定）
TURN_ANGLE = 45     # 左回転角度（度）

def main():
    et = ETRobot()
    try:
        print(f"turn_left(degree={TURN_ANGLE}, power={ARC_POWER})で45度左回転テスト")
        et.turn_left(degree=TURN_ANGLE, power=ARC_POWER)
        time.sleep(3)
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
