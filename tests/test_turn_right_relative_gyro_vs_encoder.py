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

# ジャイロzオフセット（静止時の平均値）を取得
offset_samples = []
for _ in range(30):
    _, _, z = et.get_gyro_xyz()
    offset_samples.append(z)
    time.sleep(0.01)
gyro_offset = sum(offset_samples) / len(offset_samples)
print(f"[INFO] ジャイロzオフセット: {gyro_offset}")

# 積分開始
angle_sum = 0.0
finished = False
start_time = time.time()
TIMEOUT = 5.0  # 秒
prev_time = time.time()
while not finished:
    now = time.time()
    dt = now - prev_time
    prev_time = now
    left_position = et.get_spike_status().motors["A"].relative_position or 0
    _, _, z_now = et.get_gyro_xyz()
    # オフセット補正して積分
    angle_sum += (z_now - gyro_offset) * dt
    print(f"gyro_z={z_now}, angle_sum={angle_sum}, encoder={left_position}, dt={dt}")
    # 角度が-360度以下になったら停止（右旋回前提）
    if angle_sum <= -360:
        print(f"speed={SPEED}, angle_sum={angle_sum}, encoder={left_position}")
        results.append((SPEED, angle_sum, left_position))
        et.brake()
        finished = True
    elif now - start_time > TIMEOUT:
        print("[TIMEOUT] 強制停止")
        print(f"speed={SPEED}, angle_sum={angle_sum}, encoder={left_position}")
        results.append((SPEED, angle_sum, left_position))
        et.brake()
        finished = True
    else:
        et.set_motor_forward_speed(int(SPEED), 0)
    time.sleep(0.005)
time.sleep(1)



print("\n=== Summary ===")
print("speed,gyro_z_diff,encoder")
for speed, z_diff, left_position in results:
    print(f"{speed},{z_diff},{left_position}")

et.stop()
