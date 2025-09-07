import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from nnspike.unit.etrobot import ETRobot
from nnspike.unit.action_chain import ActionChain

# テスト用速度（1つだけでOK）
SPEED = 80

# 結果記録用
results = []

# ロボット初期化
et = ETRobot()
action_chain = ActionChain(et, course="right", course_type="upper")


print(f"\n--- 360度右旋回テスト speed={SPEED} ---")
# 初期化
et.set_motor_relative_position(0, 0)
time.sleep(0.5)
# ジャイロz初期値
_, _, z_start = et.get_gyro_xyz()
# 旋回開始
finished = False
while not finished:
    left_position = et.get_spike_status().motors["A"].relative_position or 0
    _, _, z_now = et.get_gyro_xyz()
    z_diff = z_now - z_start
    # エンコーダ値が約860（90度時430の4倍）で停止
    if left_position >= 1720:  # 2回転分（1回転で860なら2回転で1720）
        print(f"speed={SPEED}, gyro_z_diff={z_diff}, encoder={left_position}")
        results.append((SPEED, z_diff, left_position))
        et.brake()
        finished = True
    else:
        et.set_motor_forward_speed(int(SPEED), 0)
    time.sleep(0.02)
time.sleep(1)



print("\n=== Summary ===")
print("speed,gyro_z_diff,encoder")
for speed, z_diff, left_position in results:
    print(f"{speed},{z_diff},{left_position}")

et.stop()
