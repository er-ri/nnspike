#!/usr/bin/env python3
"""
Neural Network Spike Robot Control - Simplified Speed Control

This script controls a line-following robot using neural network predictions
with a simplified speed control algorithm for easier tuning:

Speed Tuning Parameters:
- BASE_SPEED: Base speed for straight lines (start here)

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
"""
import argparse
import math
import pickle
import socket
import struct
import time

import cv2
import torch

from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, RELATIVE_POSITION_SCALE, ROI_CNN, Mode
from nnspike.models import NvidiaModel
from nnspike.unit import ETRobot, avoid_obstacle
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info
from scripts.utils import process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 35  # Base speed for straight lines (adjust this first)
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


def main(model_path, record_sensor_data=False, save_camera_video=False, send_video_stream=False):
    # Initialize model
    model = NvidiaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime()) if (record_sensor_data or save_camera_video) else None

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()

    # Initialize video writer conditionally
    video_writer = None
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
        video_filename = f"storage/videos/{TIMESTAMP}_picamera.avi"
        video_writer = cv2.VideoWriter(
            filename=video_filename,
            fourcc=fourcc,
            fps=30,
            frameSize=(640, 480),
        )  # Socket connection for sending camera capture (only if enabled)
    client_socket = None
    if send_video_stream:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((HOST_IP_ADDRESS, 8485))
            print(f"Connected to host PC at {HOST_IP_ADDRESS}:8485 for video streaming")
        except Exception as e:
            print(f"Warning: Could not connect to host PC for video streaming: {e}")
            client_socket = None  # Initialization
    et = ETRobot()

    pid = PIDController(
        Kp=50,
        Ki=0,
        Kd=5,
        setpoint=0,
        output_limits=(-100, 100),  # Direct radian limits for steering correction
    )

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

            roi_area = process_image(image=frame.copy(), device=device, roi=(x1, y1, x2, y2))

            rel_pos_a = status.motors["A"].relative_position
            rel_pos_b = status.motors["B"].relative_position
            rel_pos_a = rel_pos_a if rel_pos_a is not None else 0
            rel_pos_b = rel_pos_b if rel_pos_b is not None else 0
            relative_pos_value = abs(rel_pos_a) + abs(rel_pos_b)
            relative_position = abs(relative_pos_value / RELATIVE_POSITION_SCALE) if relative_pos_value is not None else 0.0
            relative_position = torch.tensor(relative_position, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(roi_area, relative_position)

            # ToDO: Use the 'Mode' output to determine the driving mode
            prob, mode = torch.max(outputs[0], dim=1)
            prob_value = round(prob[0].item(), 2)
            mode_value = mode.item()  # Convert to Python integer
            roi_center_x = (x1 + x2) / 2
            predicted_x = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()

            if mode_value == Mode.OBSTACLE_AVOIDANCE:
                # Invoke obstacle avoidance behavior
                avoid_obstacle(et)
                continue

            target_x = predicted_x  # Default to predicted x if no edge following mode is set

            offset_pixels = target_x - roi_center_x  # Calculate attitude angle using camera geometry
            theta = calculate_attitude_angle(offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS)  # Use base speed consistently

            steering_correction = pid.update(theta)

            left_speed = BASE_SPEED - steering_correction
            right_speed = BASE_SPEED + steering_correction

            et.set_motor_speed(left_speed=int(left_speed), right_speed=int(right_speed))

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status())  # Send driving information for the real-time inspection

            if send_video_stream and client_socket is not None:
                info = dict()
                info["target_x"], info["offset_y"] = target_x, OFFSET_Y
                info["text"] = {
                    "theta_deg": math.degrees(theta),
                    "relative_position": relative_position.item(),
                    "steering_correction": steering_correction,
                    "left_speed": int(left_speed),
                    "right_speed": int(right_speed),
                    "mode": mode_value,
                    "probability": prob_value,
                }

                gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                gray = draw_driving_info(gray, info, (x1, y1, x2, y2))

                try:
                    ret, buffer = cv2.imencode(".jpg", gray)
                    img_encoded = buffer.tobytes()
                    data = pickle.dumps(img_encoded)
                    client_socket.sendall(struct.pack("L", len(data)) + data)
                except Exception as e:
                    print(f"Socket error: {e}")
                    break

    finally:
        et.stop()
        cap.release()

        # Clean up socket connection if it was used
        if send_video_stream and client_socket is not None:
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
        description="Run the robot with optional sensor recording and video saving",
        epilog='Example usage: python run.py --model-path "./storage/models/model_left.pth"',
    )
    parser.add_argument("--model-path", required=True, help="Path to the trained model file")
    parser.add_argument("--record-sensor", action="store_true", help="Record sensor data to file")
    parser.add_argument("--save-video", action="store_true", help="Save camera video to file")
    parser.add_argument("--send-video", action="store_true", help="Send video stream to host PC")

    args = parser.parse_args()

    main(model_path=args.model_path, record_sensor_data=args.record_sensor, save_camera_video=args.save_video, send_video_stream=args.send_video)
