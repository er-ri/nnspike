import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from nnspike.unit.etrobot import ETRobot
# テスト用速度リスト（安定推奨値）
SPEED_LIST = [40]

def main():
    et = ETRobot()
    SPEED = 40
    ENCODER_DEG_PER_COUNT = 0.45  # 実験値
    # 90度テスト: right_enc=200で停止
    et.set_motor_relative_position(0, 0)
    time.sleep(0.5)
    print("\n--- [90度テスト] right_enc=200で停止 ---")
    angle_sum = 0.0
    prev_time = time.time()
    et.set_motor_speed(-SPEED, SPEED)
    while True:
        gyro_z = et.get_gyro_xyz()[2]
        left_enc, right_enc = et.get_motor_position()
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        angle_sum += gyro_z * dt / 20.0
        encoder_angle = right_enc * ENCODER_DEG_PER_COUNT
        print(f"gyro_z={gyro_z}, angle_sum={angle_sum:.2f}, left_enc={left_enc}, right_enc={right_enc}, encoder_angle={encoder_angle:.2f}, dt={dt}")
        if abs(right_enc) >= 200:
            et.brake()
            break
    print(f"[RESULT] 90度: right_enc={right_enc}, encoder_angle={encoder_angle:.2f}, angle_sum(z値)={angle_sum:.2f}")

    # 45度テスト: right_enc=100で停止
    et.set_motor_relative_position(0, 0)
    time.sleep(0.5)
    print("\n--- [45度テスト] right_enc=100で停止 ---")
    angle_sum = 0.0
    prev_time = time.time()
    et.set_motor_speed(-SPEED, SPEED)
    while True:
        gyro_z = et.get_gyro_xyz()[2]
        left_enc, right_enc = et.get_motor_position()
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        angle_sum += gyro_z * dt / 20.0
        encoder_angle = right_enc * ENCODER_DEG_PER_COUNT
        print(f"gyro_z={gyro_z}, angle_sum={angle_sum:.2f}, left_enc={left_enc}, right_enc={right_enc}, encoder_angle={encoder_angle:.2f}, dt={dt}")
        if abs(right_enc) >= 100:
            et.brake()
            break
    print(f"[RESULT] 45度: right_enc={right_enc}, encoder_angle={encoder_angle:.2f}, angle_sum(z値)={angle_sum:.2f}")

