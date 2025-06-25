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
ARC_POWER = 40
ARC_DURATION = 2.0  # 弧を描く時間（仮: 2m相当、要調整）
FORWARD_POWER = 40
FORWARD_DISTANCE = 2.0  # m単位
SPEED_MPS = 0.5  # 仮: 0.5m/s（要実測で調整）
FORWARD_DURATION = FORWARD_DISTANCE / SPEED_MPS

def main():
    et = ETRobot()
    try:
        # 右モーターのみ180度回転（左モーターは0度）
        print("左モーター: 0度、右モーター: 180度 正回転（右のみ45度左方向回転相当）")
        et.set_motor_degrees(left_degrees=0, right_degrees=180)
        time.sleep(3)
        # 右カーブで前進（右モーター弱・左モーター強）
        print("右カーブ: power=40, duration=2.0秒で前進（右モーター弱・左モーター強）")
        et.move_right_arc(duration=ARC_DURATION, power=ARC_POWER)
        #time.sleep(3)
        et.brake()
        time.sleep(3)
    except KeyboardInterrupt:
        print("中断されました。停止します。")
        et.brake()
    finally:
        et.stop()

if __name__ == "__main__":
    main()
