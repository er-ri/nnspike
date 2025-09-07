import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from nnspike.unit.etrobot import ETRobot
# テスト用速度リスト（安定推奨値）
SPEED_LIST = [40]

def main():
    et = ETRobot()
    results = []
    for test_angle, label in [(-90, "90度"), (-45, "45度")]:
        # 各テスト角度の最初でエンコーダリセット
        et.set_motor_relative_position(0, 0)
        time.sleep(0.5)
        for SPEED in SPEED_LIST:
            print(f"\n--- {label}右旋回テスト speed={SPEED} ---")
            GYRO_SCALE = 1.0
            print(f"[INFO] GYRO_SCALE={GYRO_SCALE}")
            angle_sum = 0.0
            finished = False
            start_time = time.time()
            TIMEOUT = 5.0  # 秒
            prev_time = time.time()
            time.sleep(0.2)
            ENCODER_DEG_PER_COUNT = 0.45  # 物理基準: エンコーダ値-200で90度
            # 目標エンコーダ値（右モーター）を基準値で設定
            if test_angle == -90:
                target_encoder = -200
            elif test_angle == -45:
                target_encoder = -100
            else:
                target_encoder = int(test_angle / 0.45)
            while not finished:
                now = time.time()
                dt = now - prev_time
                prev_time = now
                left_position = et.get_spike_status().motors["A"].relative_position or 0
                right_position = et.get_spike_status().motors["B"].relative_position or 0
                _, _, z_now = et.get_gyro_xyz()
                angle_sum += z_now * dt
                angle_deg = angle_sum * GYRO_SCALE
                encoder_angle = right_position * ENCODER_DEG_PER_COUNT
                print(f"gyro_z={z_now}, angle_sum={angle_sum}, angle_deg={angle_deg}, left_enc={left_position}, right_enc={right_position}, encoder_angle={encoder_angle:.2f}, dt={dt}")
                # エンコーダ値が目標値以下になったら停止（右旋回前提）
                if right_position <= target_encoder:
                    print(f"[OK] speed={SPEED}, encoder_angle={encoder_angle:.2f}, right_enc={right_position}, target_enc={target_encoder}, angle_deg={angle_deg:.2f}")
                    results.append((SPEED, angle_deg, right_position, GYRO_SCALE))
                    et.brake()
                    et.set_motor_forward_speed(0, 0)
                    finished = True
                elif angle_deg <= -400:
                    print("[SAFETY] 角度-400度超えで強制停止")
                    print(f"speed={SPEED}, angle_sum={angle_sum}, angle_deg={angle_deg}, left_enc={left_position}, right_enc={right_position}, GYRO_SCALE={GYRO_SCALE}")
                    results.append((SPEED, angle_deg, right_position, GYRO_SCALE))
                    et.brake()
                    et.set_motor_forward_speed(0, 0)
                    finished = True
                elif now - start_time > TIMEOUT:
                    print("[TIMEOUT] 強制停止")
                    print(f"speed={SPEED}, angle_sum={angle_sum}, angle_deg={angle_deg}, left_enc={left_position}, right_enc={right_position}, GYRO_SCALE={GYRO_SCALE}")
                    results.append((SPEED, angle_deg, right_position, GYRO_SCALE))
                    et.brake()
                    et.set_motor_forward_speed(0, 0)
                    finished = True
                else:
                    et.set_motor_speed(int(SPEED), -int(SPEED))
                time.sleep(0.005)
            elapsed = time.time() - start_time
            print(f"[SUMMARY] {label} speed={SPEED} time={elapsed:.2f}s right_encoder={right_position} encoder_angle={encoder_angle:.2f}deg gyro_angle={angle_deg:.2f}deg ENCODER_DEG_PER_COUNT={ENCODER_DEG_PER_COUNT:.3f}")
            time.sleep(2)

    et.stop()

if __name__ == "__main__":
    main()