#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

This script controls a line-following robot using OpenCV computer vision
instead of neural network predictions. It uses the steer_by_camera function
to detect the line centroid and follows it using PID control.

Speed Tuning Parameters:
- BASE_POWER: Base power for straight lines (start here)
- TURN_REDUCTION_FACTOR: Speed reduction in turns (0.0-1.0)
- TURN_THRESHOLD: Pixel offset threshold to detect turns

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
"""

# ==== ユーザー調整用パラメータ（ここだけ編集すればOK） ====
ROI_OPENCV = (150, 300, 490, 400)  # 必要に応じて変更
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
BASE_POWER = 30         # 基本パワー（直線時以外、カーブ時の基準）
STRAIGHT_POWER = 50     # 直線時専用のパワー（カーブでなく黒ライン上のみ）
CURVE_POWER = 20        # カーブ時の最低パワー（必要に応じて使用）
CURVE_THRESHOLD_DEG = 3.0      # カーブ判定閾値（度数法, 例: 3度）
SENSITIVITY = 0.4           # ピクセル→theta変換感度
MAX_STEERING_POWER_DIFF = 40   # 最大旋回時の左右パワー差（%）
MAX_STEERING_THETA_DEG = 30    # 最大旋回角（度数法, 例: 30度）
# 黒ライン判定の閾値（より安全側に余裕を持たせる）
BLACK_LINE_REFLECTED_THRESHOLD = 40  # 反射光R: 40以下なら黒
BLACK_LINE_COLOR_THRESHOLD = 150     # color: 150以下なら黒
# ================================================

import cv2
import math
import time
import socket
import pickle
import struct
import argparse
import numpy as np
from nnspike.unit import ETRobot
from nnspike.utils import (
    steer_by_camera,
    draw_driving_info,
    PIDController,
    SensorRecorder,
    calculate_adaptive_speed,
    calculate_theta_from_pixels,
)

# User defined constants
x1, y1, x2, y2 = ROI_OPENCV  # Region of Interest for OpenCV processing


# Socket connection settings
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


# AsyncSensorReader クラスを削除 - メインループで直接センサー値を取得するように変更


def run_initial_sensor_test(et, test_count=5, delay=0.2):
    """SPIKEの初期センサーテストを簡素に実行"""
    print("初期センサーテスト...")
    for i in range(test_count):
        test_status = et.get_spike_status()
        if test_status and test_status.sensors:
            color = test_status.sensors.color
            dist = test_status.sensors.distance
            color_str = f"R:{color.reflected} A:{color.ambient} C:{color.color}" if color else "N/A"
            dist_str = f"{dist}cm" if dist is not None else "N/A"
            print(f"{i+1}: OK  Color={color_str}  US={dist_str}")
        else:
            print(f"{i+1}: spike_status取得失敗")
        time.sleep(delay)
    print("初期センサーテスト完了")


def initialize_system(record_sensor_data, save_camera_video):
    """ロボット・PID・センサーレコーダ・ビデオ・ソケット等の初期化をまとめて行う"""
    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = (
        time.strftime("%Y%m%d%H%M%S", time.localtime())
        if (record_sensor_data or save_camera_video)
        else None
    )

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()

    # Initialize video writer conditionally
    video_writer = None
    video_filename = None
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(IMAGE_WIDTH, IMAGE_HEIGHT),
        )
    # Socket connection for sending camera capture
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP_ADDRESS, 8485))

    # Initialize robot and PID controller
    et = ETRobot()
    pid = PIDController(
        Kp=3,  # Restored to ensure sufficient turning power
        Ki=0,
        Kd=0.2,  # Increased derivative term to reduce oscillation
        setpoint=0,
        output_limits=(-0.25, 0.25),  # Direct radian limits for steering correction
    )
    print("メインループで直接センサー値を取得します")

    # --- アームを1秒上げて1秒下げる処理を追加（test_color_only.py参考） ---
    try:
        print("Arm up...")
        et.move_arm(1)  # 1 = up
        time.sleep(1.0)
        print("Arm down...")
        et.move_arm(0)  # 0 = down
        time.sleep(1.0)
    except Exception as e:
        print(f"Arm move error: {e}")

    # 初期センサーテストを関数で実行
    run_initial_sensor_test(et)

    return et, pid, sensor_recorder, video_writer, video_filename, client_socket


def send_stop_signal(et, duration=5.0):
    print(f"Sending stop signals to Spike for {duration} seconds...")
    stop_start_time = time.time()
    while time.time() - stop_start_time < duration:
        try:
            et.brake()
            time.sleep(0.1)
        except Exception as e:
            print(f"Error sending stop signal: {e}")
            break
    print("Stop signal transmission completed")


def main(record_sensor_data=False, save_camera_video=False):
    et, pid, sensor_recorder, video_writer, video_filename, client_socket = initialize_system(record_sensor_data, save_camera_video)
    time.sleep(0.5)
    et.set_motor_relative_position(left_positon=0, right_position=0)

    try:
        while et.is_running == True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break            
            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)
            
            # Process frame for steering using updated steer_by_camera function
            mx, my, offset_pixels, max_contour = steer_by_camera(frame, ROI_OPENCV)
            theta = calculate_theta_from_pixels(
                offset_pixels=offset_pixels,
                image_width=IMAGE_WIDTH,
                sensitivity=SENSITIVITY
            )
            
            # spike_statusの取得を最初にまとめる
            spike_status = et.get_spike_status()

            # カラーセンサー値取得・データ生成・黒ライン判定
            if spike_status and spike_status.sensors and spike_status.sensors.color:
                color = spike_status.sensors.color
                color_data = f"R:{color.reflected} A:{color.ambient} C:{color.color}"
                # より安全側に余裕を持たせた黒ライン判定
                ON_BLACK_LINE = (
                    color.reflected is not None and color.reflected <= BLACK_LINE_REFLECTED_THRESHOLD and
                    color.color is not None and color.color <= BLACK_LINE_COLOR_THRESHOLD
                )
            else:
                color_data = "R:N/A A:N/A C:N/A"
                ON_BLACK_LINE = False

            # Dynamic speed control using external function
            abs_theta = abs(theta)
            current_base_power = calculate_adaptive_speed(
                abs_theta=abs_theta,
                base_power=BASE_POWER,
                curve_power=CURVE_POWER,
                straight_power=STRAIGHT_POWER,
                curve_threshold=math.radians(CURVE_THRESHOLD_DEG),
                on_black_line=ON_BLACK_LINE
            )

            # Apply PID control to theta for smooth steering correction
            pid_corrected_theta = pid.update(theta)
            # 最大旋回時のパワー差を直感的に指定
            max_theta = math.radians(MAX_STEERING_THETA_DEG)
            power_adjustment = int((pid_corrected_theta / max_theta) * MAX_STEERING_POWER_DIFF)
            left_power = int(current_base_power - power_adjustment)
            right_power = int(current_base_power + power_adjustment)

            # 計算した左右パワーでモーターを駆動
            et.set_motor_forward_power(
                left_power=left_power,
                right_power=right_power,
            )
            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(spike_status)  # 既に取得したspike_statusを再利用
            
            # Prepare driving information for visualization
            info = dict()
            info["offset_x"], info["offset_y"] = x1 + mx, y1 + my            # メインループで直接センサーデータを取得
            # 超音波センサーデータの生成
            if spike_status and spike_status.sensors and spike_status.sensors.distance is not None:
                ultrasonic_data = f"{spike_status.sensors.distance} cm"
            else:
                ultrasonic_data = "N/A cm"
            
            info["text"] = {
                "theta_deg": f"{round(math.degrees(theta), 2)}deg",
                "pid_corrected_theta": f"{round(math.degrees(pid_corrected_theta), 2)}deg",
                "offset_pixels": f"{round(offset_pixels, 1)}px",
                "current_power": f"{round(current_base_power, 1)}%",
                "curve_detected": ("OFF_LINE" if theta == 0 else 
                                 "YES" if abs_theta > math.radians(CURVE_THRESHOLD_DEG) else "NO"),
                "on_black_line": "YES" if ON_BLACK_LINE else "NO",
                "left_power": f"{left_power}%",
                "right_power": f"{right_power}%",
                "color_sensor": color_data,
                "ultrasonic_sensor": ultrasonic_data,
                "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
            }

            # Create visualization frame
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray = draw_driving_info(gray, info, (x1, y1, x2, y2))

            # Draw contour on the visualization if found
            if max_contour is not None:
                # Adjust contour coordinates to full frame
                adjusted_contour = max_contour + np.array([x1, y1])
                cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)                # Draw centroid
                cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)

            # Send camera capture for remote monitoring
            try:
                # 変更後: JPG形式（品質80）でエンコード
                ret, buffer = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                img_encoded = buffer.tobytes()
                data = pickle.dumps(img_encoded)
                client_socket.sendall(struct.pack("L", len(data)) + data)
            except Exception as e:
                print(f"Socket error: {e}")
                break

            # ループ終了時に30ms間隔となるようsleep
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.03 - elapsed)
            #print(f"[DEBUG] loop_elapsed: {elapsed*1000:.2f} ms, sleep: {sleep_time*1000:.2f} ms")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Interrupted by user")
        send_stop_signal(et, duration=5.0)
    except Exception as e:
        print(f"Error: {e}")
        send_stop_signal(et, duration=5.0)
    finally:
        et.stop()
        cap.release()
        client_socket.close()
        if save_camera_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the OpenCV-based line following robot with optional sensor recording and video saving"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Record sensor data to file"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save camera video to file"
    )

    args = parser.parse_args()

    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_OPENCV}")
    print(f"Base power: {BASE_POWER}")
    print("Press Ctrl+C to stop")

    main(record_sensor_data=args.record_sensor, save_camera_video=args.save_video)
