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
import cv2
import math
import time
import socket
import pickle
import struct
import argparse
import asyncio
import threading
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
from nnspike.constants import (
    ROI_OPENCV,
)

# User defined constants
x1, y1, x2, y2 = ROI_OPENCV  # Region of Interest for OpenCV processing

# Simplified Speed Control Parameters (Easy to tune)
BASE_POWER = 50  # Base power for straight lines (adjust this first)
MAX_POWER = 80   # Maximum power for straight lines
CURVE_POWER = 20 # Power for curves (rapid deceleration)
CURVE_THRESHOLD = 0.0524  # Threshold to detect curves (≈3.0 degrees, adjusted boundary)

# Steering Control Parameters
STEERING_SCALE_FACTOR = 30  # Scaling factor to convert steering correction (radians) to power adjustment

# Socket connection settings
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


class AsyncSensorReader:
    """非同期でセンサー値を取得するクラス"""
    
    def __init__(self, et_robot):
        self.et_robot = et_robot
        self.color_sensor_data = "R:N/A A:N/A C:N/A"
        self.ultrasonic_sensor_data = "N/A cm"
        self.running = False
        self.thread = None
        
    def start(self):
        """センサー読み取りを開始"""
        self.running = True
        self.thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """センサー読み取りを停止"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _sensor_loop(self):
        """0.2秒間隔でセンサー値を取得するループ"""
        print("AsyncSensorReader: センサー読み取りループを開始しました")
        
        loop_count = 0
        previous_color = ""
        previous_ultrasonic = ""
        
        while self.running:
            try:
                # spike_statusを一度取得
                spike_status = self.et_robot.get_spike_status()
                  # カラーセンサーデータを更新
                if spike_status and spike_status.sensors and spike_status.sensors.color:
                    color = spike_status.sensors.color
                    new_color_data = f"R:{color.reflected} A:{color.ambient} C:{color.color}"
                    
                    # より詳細な値変化も検知
                    if new_color_data != self.color_sensor_data:
                        print(f"AsyncSensorReader: カラーセンサー変化 - {self.color_sensor_data} → {new_color_data}")
                    
                    self.color_sensor_data = new_color_data
                else:
                    self.color_sensor_data = "R:N/A A:N/A C:N/A"
                
                # 超音波センサーデータを更新
                if spike_status and spike_status.sensors and spike_status.sensors.distance is not None:
                    new_ultrasonic_data = f"{spike_status.sensors.distance}cm"
                    
                    # より詳細な値変化も検知
                    if new_ultrasonic_data != self.ultrasonic_sensor_data:
                        print(f"AsyncSensorReader: 超音波センサー変化 - {self.ultrasonic_sensor_data} → {new_ultrasonic_data}")
                    
                    self.ultrasonic_sensor_data = new_ultrasonic_data
                else:
                    self.ultrasonic_sensor_data = "N/A cm"
                
                # センサー値が変化したときのみログ出力
                if (self.color_sensor_data != previous_color or 
                    self.ultrasonic_sensor_data != previous_ultrasonic):
                    print(f"AsyncSensorReader: 値変化検知 - Color: {previous_color} → {self.color_sensor_data}, Ultrasonic: {previous_ultrasonic} → {self.ultrasonic_sensor_data}")
                    previous_color = self.color_sensor_data
                    previous_ultrasonic = self.ultrasonic_sensor_data
                  # 10秒ごと（50ループごと）に詳細ログ出力
                loop_count += 1
                if loop_count % 50 == 0:
                    # spike_statusの詳細情報も出力
                    if spike_status and spike_status.sensors:
                        if spike_status.sensors.color:
                            color = spike_status.sensors.color
                            print(f"AsyncSensorReader: 生データ確認 - カラー reflected={color.reflected}, ambient={color.ambient}, color={color.color}")
                        if spike_status.sensors.distance is not None:
                            print(f"AsyncSensorReader: 生データ確認 - 超音波 distance={spike_status.sensors.distance}")
                    print(f"AsyncSensorReader: 定期更新 - Color={self.color_sensor_data}, Ultrasonic={self.ultrasonic_sensor_data}")
                    print(f"AsyncSensorReader: ループカウント={loop_count}, spike_status取得成功={spike_status is not None}")
                    
            except Exception as e:
                # エラー時はデフォルト値を設定
                print(f"AsyncSensorReader: エラー - {e}")
                self.color_sensor_data = "R:N/A A:N/A C:N/A"
                self.ultrasonic_sensor_data = "N/A cm"
                
            time.sleep(0.2)  # 0.2秒間隔
        
        print("AsyncSensorReader: センサー読み取りループを終了しました")
            
    def get_color_sensor_data(self):
        """現在のカラーセンサーデータを取得"""
        return self.color_sensor_data
        
    def get_ultrasonic_sensor_data(self):
        """現在の超音波センサーデータを取得"""
        return self.ultrasonic_sensor_data


def main(record_sensor_data=False, save_camera_video=False):
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
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(640, 480),
        )    # Socket connection for sending camera capture
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(
        (HOST_IP_ADDRESS, 8485)
    )
    
    # Initialize robot and PID controller
    et = ETRobot()
    pid = PIDController(
        Kp=3,  # Restored to ensure sufficient turning power
        Ki=0,
        Kd=0.2,  # Increased derivative term to reduce oscillation
        setpoint=0,
        output_limits=(-0.25, 0.25),  # Direct radian limits for steering correction
    )    # 非同期センサーリーダーを初期化して開始
    sensor_reader = AsyncSensorReader(et)
    sensor_reader.start()
    print("AsyncSensorReader started")
    
    # 初期センサーテスト
    print("初期センサーテストを実行中...")
    for i in range(5):
        test_status = et.get_spike_status()
        if test_status and test_status.sensors:
            print(f"テスト{i+1}: spike_status OK")
            if test_status.sensors.color:
                color = test_status.sensors.color
                print(f"  カラーセンサー: R={color.reflected}, A={color.ambient}, C={color.color}")
            else:
                print("  カラーセンサー: データなし")
            if test_status.sensors.distance is not None:
                print(f"  超音波センサー: {test_status.sensors.distance}cm")
            else:
                print("  超音波センサー: データなし")
        else:
            print(f"テスト{i+1}: spike_status取得失敗")
        time.sleep(0.2)
    print("初期センサーテスト完了")
    
    # Time-based acceleration tracking
    straight_line_start_time = None
    acceleration_duration = 1.0  # 1 second to reach MAX_POWER
    
    time.sleep(0.5)
    et.set_motor_relative_position(left_positon=0, right_position=0)

    try:
        while et.is_running == True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break            
            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)
            
            # Process frame for steering using updated steer_by_camera function
            mx, my, offset_pixels, max_contour = steer_by_camera(frame, ROI_OPENCV)
            
            # Calculate theta (attitude angle in radians): +right deviation, -left deviation, 0=centered
            theta = calculate_theta_from_pixels(
                offset_pixels=offset_pixels,
                image_width=640,
                sensitivity=0.4
            )
            
            # Dynamic speed control using external function
            current_time = time.time()
            abs_theta = abs(theta)
            current_base_power, straight_line_start_time = calculate_adaptive_speed(
                abs_theta=abs_theta,
                current_time=current_time,
                straight_line_start_time=straight_line_start_time,
                base_power=BASE_POWER,
                max_power=MAX_POWER,
                curve_power=CURVE_POWER,
                curve_threshold=CURVE_THRESHOLD,
                acceleration_duration=1.0
            )

            # Apply PID control to theta for smooth steering correction
            pid_corrected_theta = pid.update(theta)
            
            # pid_corrected_theta is in radians, convert to power adjustment
            power_adjustment = int(pid_corrected_theta * STEERING_SCALE_FACTOR)
            left_power = int(current_base_power - power_adjustment)
            right_power = int(current_base_power + power_adjustment)

            et.set_motor_forward_power(
                left_power=left_power,
                right_power=right_power,
            )
              # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status())
            
            # Prepare driving information for visualization
            info = dict()
            info["offset_x"], info["offset_y"] = x1 + mx, y1 + my            # センサーデータを取得して表示
            color_data = sensor_reader.get_color_sensor_data()
            ultrasonic_data = sensor_reader.get_ultrasonic_sensor_data()
            
            # デバッグ用: センサー値をコンソールに出力（頻度を下げる）
            current_second = int(time.time())
            if not hasattr(main, 'last_debug_second') or main.last_debug_second != current_second:
                main.last_debug_second = current_second
                if current_second % 2 == 0:  # 2秒ごとに出力
                    print(f"メインループ: Color={color_data}, Ultrasonic={ultrasonic_data}")
                    # info["text"]の値も確認
                    print(f"メインループ: info['text']の color_sensor と ultrasonic_sensor を確認中...")
            
            info["text"] = {
                "theta_deg": f"{round(math.degrees(theta), 2)}deg",
                "pid_corrected_theta": f"{round(math.degrees(pid_corrected_theta), 2)}deg",
                "offset_pixels": f"{round(offset_pixels, 1)}px",
                "current_power": f"{round(current_base_power, 1)}%",
                "curve_detected": ("OFF_LINE" if theta == 0 else 
                                 "YES" if abs_theta > CURVE_THRESHOLD else "NO"),
                "acceleration_time": f"{(round(current_time - straight_line_start_time, 1) if straight_line_start_time is not None else 0.0)}s",
                "left_power": f"{left_power}%",
                "right_power": f"{right_power}%",
                "color_sensor": color_data,
                "ultrasonic_sensor": ultrasonic_data,
                "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
            }
            
            # デバッグ用: info["text"]のセンサー値を確認
            if current_second % 2 == 0 and hasattr(main, 'last_debug_second') and main.last_debug_second == current_second:
                print(f"メインループ: info['text']['color_sensor'] = {info['text']['color_sensor']}")
                print(f"メインループ: info['text']['ultrasonic_sensor'] = {info['text']['ultrasonic_sensor']}")

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
                ret, buffer = cv2.imencode(".png", gray)
                img_encoded = buffer.tobytes()
                data = pickle.dumps(img_encoded)
                client_socket.sendall(struct.pack("L", len(data)) + data)
            except Exception as e:
                print(f"Socket error: {e}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
        print("Sending stop signals to Spike for 10 seconds...")
        
        # 10秒間停止信号を送信
        stop_start_time = time.time()
        while time.time() - stop_start_time < 10.0:
            try:
                et.brake()  # モーター停止信号を送信
                time.sleep(0.1)  # 100ms間隔で送信
            except Exception as e:
                print(f"Error sending stop signal: {e}")
                break
        
        print("Stop signal transmission completed")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Sending stop signals to Spike for 10 seconds...")
        
        # エラー時も10秒間停止信号を送信
        stop_start_time = time.time()
        while time.time() - stop_start_time < 10.0:
            try:
                et.brake()  # モーター停止信号を送信
                time.sleep(0.1)  # 100ms間隔で送信
            except Exception as e:
                print(f"Error sending stop signal: {e}")
                break
        
        print("Stop signal transmission completed")
        
    finally:
        # センサーリーダーを停止
        sensor_reader.stop()
        
        # Cleanup
        et.stop()
        cap.release()
        client_socket.close()

        # Clean up video writer if it was used
        if save_camera_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")

        # Clean up sensor recorder if it was used
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
