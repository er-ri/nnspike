#!/usr/bin/env python3
"""
Neural Network Spike Robot Control - Simplified Speed Control

This script controls a line-following robot using neural network predictions
with a simplified speed control algorithm for easier tuning:

Speed Tuning Parameters:
- BASE_POWER: Base power for straight lines (start here)
- TURN_REDUCTION_FACTOR: Speed reduction in turns (0.0-1.0)
- TURN_THRESHOLD: Angle threshold to detect turns (radians)

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
"""
import cv2
import math
import time
import torch
import socket
import pickle
import struct
import argparse
from nnspike.unit import ETRobot
from nnspike.utils import (
    draw_driving_info,
    PIDController,
    SensorRecorder,
)
from nnspike.utils.control import (
    calculate_attitude_angle,
    calculate_differential_steering,
)
from nnspike.constants import (
    ROI_CNN,
    CAMERA_HEIGHT,
    CAMERA_FOCAL_LENGTH_PIXELS,
    WHEELBASE,
)
from scripts.utils import load_and_prepare_model, process_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest
LOW_POWER = 35
HIGH_POWER = 50

# Simplified Speed Control Parameters (Easy to tune)
BASE_POWER = 50  # Base power for straight lines (adjust this first)
TURN_REDUCTION_FACTOR = 0.7  # Reduce speed in turns (0.0-1.0, lower = slower in turns)
TURN_THRESHOLD = 0.1  # Angle threshold to detect turns (radians, ~5.7 degrees)
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)

course = "left"  # "left" or "right"
model_paths = [
    f"./storage/models/{course}_interval1_0601.pth",
    f"./storage/models/{course}_interval2_0601.pth",
    f"./storage/models/{course}_interval3_0601.pth",
]

models = [load_and_prepare_model(path, device) for path in model_paths]

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


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
        )  # Socket connection for sending camera capture
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP_ADDRESS, 8485))  # Initialization
    et = ETRobot()
    pid = PIDController(
        Kp=20,
        Ki=0,
        Kd=5,
        setpoint=0,
        output_limits=(-0.30, 0.30),  # Direct radian limits for steering correction
    )

    time.sleep(0.5)

    et.set_motor_relative_position(left_positon=0, right_position=0)
    interval_idx = 0

    try:
        while et.is_running == True:

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)

            roi_area = process_image(image=frame.copy(), device=device, roi=ROI_CNN)

            # Todo: stage determination
            interval_idx = 0

            with torch.no_grad():
                output = models[interval_idx](
                    roi_area
                )  # Calculate pixel offset from center
            roi_center_x = (x1 + x2) / 2
            offset_x = (x1 + output * (x2 - x1))[0][0].detach().item()
            offset_pixels = (
                offset_x - roi_center_x
            )  # Calculate attitude angle using camera geometry
            theta = calculate_attitude_angle(
                offset_pixels, y2, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS
            )

            # Simple speed control: reduce speed in turns for easier tuning
            abs_theta = abs(theta)
            if abs_theta > TURN_THRESHOLD:  # In a turn
                current_base_power = BASE_POWER * TURN_REDUCTION_FACTOR
            else:  # Going straight
                current_base_power = (
                    BASE_POWER  # Use PID controller with attitude angle
                )
            steering_correction = pid.update(theta)

            # Apply steering with geometric approach
            left_power, right_power = calculate_differential_steering(
                theta + steering_correction, current_base_power, WHEELBASE
            )

            et.set_motor_forward_power(
                left_power=int(max(0, min(100, left_power))),
                right_power=int(max(0, min(100, right_power))),
            )

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(
                    et.get_spike_status()
                )  # Send driving information for the real-time inspection
            info = dict()
            info["offset_x"], info["offset_y"] = offset_x, 250
            info["text"] = {
                "theta_deg": math.degrees(theta),
                "steering_correction": steering_correction,
                "base_power": int(current_base_power),
                "left_power": int(left_power),
                "right_power": int(right_power),
            }

            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray = draw_driving_info(gray, info, (x1, y1, x2, y2))

            # Send camera capture
            try:
                ret, buffer = cv2.imencode(".png", gray)
                img_encoded = buffer.tobytes()
                data = pickle.dumps(img_encoded)
                client_socket.sendall(struct.pack("L", len(data)) + data)
            except Exception:
                break

    finally:
        et.stop()
        cap.release()

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
        description="Run the robot with optional sensor recording and video saving"
    )
    parser.add_argument(
        "--record-sensor", action="store_true", help="Record sensor data to file"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save camera video to file"
    )

    args = parser.parse_args()

    main(record_sensor_data=args.record_sensor, save_camera_video=args.save_video)
