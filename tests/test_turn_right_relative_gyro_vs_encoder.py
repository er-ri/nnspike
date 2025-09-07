import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from nnspike.unit.etrobot import ETRobot

# テスト用速度リスト（安定推奨値）
SPEED_LIST = [40]

# 結果記録用
results = []

# ロボット初期化
et = ETRobot()

for test_angle, label in [(-90, "90度"), (-180, "180度")]:
    for SPEED in SPEED_LIST:
        print(f"\n--- {label}右旋回テスト speed={SPEED} ---")
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

        # スケールファクタ初期値
        # 角度ごとにGYRO_SCALEを調整
        if test_angle == -90:
            GYRO_SCALE = 4.0  # 90度用（実機ログで調整済み）
        else:
            GYRO_SCALE = 7.2  # 180度用（実機ログで調整済み）
        print(f"[INFO] GYRO_SCALE={GYRO_SCALE}")
        angle_sum = 0.0
        finished = False
        start_time = time.time()
        TIMEOUT = 5.0  # 秒
        prev_time = time.time()
        et.set_motor_relative_position(0, 0)
        time.sleep(0.2)
        while not finished:
            now = time.time()
            dt = now - prev_time
            prev_time = now
            left_position = et.get_spike_status().motors["A"].relative_position or 0
            _, _, z_now = et.get_gyro_xyz()
            # オフセット補正して積分
            angle_sum += (z_now - gyro_offset) * dt
            angle_deg = angle_sum * GYRO_SCALE
            print(f"gyro_z={z_now}, angle_sum={angle_sum}, angle_deg={angle_deg}, encoder={left_position}, dt={dt}")
            # 角度がtest_angle以下になったら停止（右旋回前提）
            if angle_deg <= test_angle:
                print(f"[OK] speed={SPEED}, angle_sum={angle_sum}, angle_deg={angle_deg}, encoder={left_position}, GYRO_SCALE={GYRO_SCALE}")
                results.append((SPEED, angle_deg, left_position, GYRO_SCALE))
                et.brake()
                et.set_motor_forward_speed(0, 0)
                finished = True
            elif angle_deg <= -400:
                print("[SAFETY] 角度-400度超えで強制停止")
                print(f"speed={SPEED}, angle_sum={angle_sum}, angle_deg={angle_deg}, encoder={left_position}, GYRO_SCALE={GYRO_SCALE}")
                results.append((SPEED, angle_deg, left_position, GYRO_SCALE))
                et.brake()
                et.set_motor_forward_speed(0, 0)
                finished = True
            elif now - start_time > TIMEOUT:
                print("[TIMEOUT] 強制停止")
                print(f"speed={SPEED}, angle_sum={angle_sum}, angle_deg={angle_deg}, encoder={left_position}, GYRO_SCALE={GYRO_SCALE}")
                results.append((SPEED, angle_deg, left_position, GYRO_SCALE))
                et.brake()
                et.set_motor_forward_speed(0, 0)
                finished = True
            else:
                et.set_motor_speed(int(SPEED), -int(SPEED))
            time.sleep(0.005)
        time.sleep(2)  # インターバル2秒

et.stop()