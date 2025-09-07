import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from nnspike.unit.etrobot import ETRobot
from nnspike.unit.action_chain import ActionChain

# テスト用速度リスト
SPEED_LIST = list(range(55, 76))  # 55〜75を1刻みでテスト

# 結果記録用
results = []

# ロボット初期化
et = ETRobot()
action_chain = ActionChain(et, course="right", course_type="upper")

for speed in SPEED_LIST:
    print(f"\n--- Testing speed={speed} ---")
    # 初期化
    et.set_motor_relative_position(0, 0)
    time.sleep(0.5)
    # ジャイロz初期値
    _, _, z_start = et.get_gyro_xyz()
    # 旋回開始
    finished = False
    while not finished:
        # 左モーターAの相対位置
        left_position = et.get_spike_status().motors["A"].relative_position or 0
        # 現在のジャイロz
        _, _, z_now = et.get_gyro_xyz()
        # 430超えたら記録
        if abs(left_position) > 430:
            z_diff = z_now - z_start
            print(f"speed={speed}, encoder=430, gyro_z_diff={z_diff}")
            results.append((speed, z_diff))
            et.brake()
            finished = True
        else:
            et.set_motor_forward_speed(speed, 0)
        time.sleep(0.02)
    time.sleep(1)

print("\n=== Summary ===")
print("speed,gyro_z_diff")
for speed, z_diff in results:
    print(f"{speed},{z_diff}")

et.stop()
