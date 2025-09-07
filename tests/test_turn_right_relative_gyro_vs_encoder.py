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

def main():
    for test_angle, label in [(-90, "90度"), (-45, "45度")]:
        # 各テスト角度の最初でエンコーダリセット
        et.set_motor_relative_position(0, 0)
        time.sleep(0.5)
    # et.reset_gyro()  # ジャイロリセット（ETRobotに未実装のため呼び出さない）
    # slot側でalign_to_model()済みのため、バイアス補正は不要

        for SPEED in SPEED_LIST:
            print(f"\n--- {label}右旋回テスト speed={SPEED} ---")
            # スケールファクタ初期値（物理挙動に合わせて暫定1.0で統一）
            GYRO_SCALE = 1.0
            print(f"[INFO] GYRO_SCALE={GYRO_SCALE}")
            angle_sum = 0.0
            finished = False
            start_time = time.time()
            TIMEOUT = 5.0  # 秒
            prev_time = time.time()
            # ここでのリセットは不要
            time.sleep(0.2)
            while not finished:
                now = time.time()
                dt = now - prev_time
                prev_time = now
                left_position = et.get_spike_status().motors["A"].relative_position or 0
                right_position = et.get_spike_status().motors["B"].relative_position or 0
                _, _, z_now = et.get_gyro_xyz()
                # オフセット補正なしで積分
                angle_sum += z_now * dt
                angle_deg = angle_sum * GYRO_SCALE
                print(f"gyro_z={z_now}, angle_sum={angle_sum}, angle_deg={angle_deg}, left_enc={left_position}, right_enc={right_position}, dt={dt}")
                # 角度がtest_angle以下になったら停止（右旋回前提）
                if angle_deg <= test_angle:
                    print(f"[OK] speed={SPEED}, angle_sum={angle_sum}, angle_deg={angle_deg}, left_enc={left_position}, right_enc={right_position}, GYRO_SCALE={GYRO_SCALE}")
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
            # テスト終了時に経過時間・エンコーダ値・ジャイロ積分値・エンコーダ角度推定値をまとめて出力
            elapsed = time.time() - start_time
            # エンコーダ値→角度変換（仮: 1回転=360度, ギア比や車輪径に応じて調整要）
            ENCODER_DEG_PER_COUNT = 0.45  # エンコーダカウントと物理角度の一致のため
            encoder_angle = right_position * ENCODER_DEG_PER_COUNT
            print(f"[SUMMARY] {label} speed={SPEED} time={elapsed:.2f}s right_encoder={right_position} encoder_angle={encoder_angle:.2f}deg gyro_angle={angle_deg:.2f}deg")
            time.sleep(2)  # インターバル2秒

    et.stop()

if __name__ == "__main__":
    main()