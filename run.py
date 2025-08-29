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
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from nnspike.constants import CAMERA_FOCAL_LENGTH_PIXELS, CAMERA_HEIGHT, OFFSET_Y, RELATIVE_POSITION_SCALE, ROI_CNN, Mode
from nnspike.unit import ActionChain, ETRobot, ModeManager, WebcamVideoStream
from nnspike.utils import PIDController, SensorRecorder, calculate_attitude_angle, draw_driving_info

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to


def main(model_path, record_sensor_data=False, save_camera_video=False, send_video_stream=False, course="left"):

    # Initialize model
    session = ort.InferenceSession(model_path)

    # Generate timestamp for consistent naming if recording is enabled
    TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime()) if (record_sensor_data or save_camera_video) else ""

    vs = WebcamVideoStream(src=0, save_video=save_camera_video, save_path=f"storage/videos/{TIMESTAMP}_picamera.avi")
    vs.start()

    # Initialize sensor recorder conditionally
    sensor_recorder = None
    if record_sensor_data:
        sensor_recorder = SensorRecorder(timestamp=TIMESTAMP)
        sensor_recorder.start_recording()

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
    action_chain = ActionChain(
        et,
        course=course,
    )
    mode_manager = ModeManager(course=course)
    pid = PIDController(
        Kp=50,
        Ki=0,
        Kd=5,
        setpoint=0,
        output_limits=(-100, 100),  # Direct radian limits for steering correction
    )

    time.sleep(0.5)

    et.set_motor_relative_position(left_positon=0, right_position=0)

    mode = Mode.FOLLOW_LEFT_EDGE if course == "left" else Mode.FOLLOW_RIGHT_EDGE

    try:
        while et.is_running == True:
            inf_start_time = time.time()

            ret, frame = vs.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            motors_relative_position = et.retrieve_motors_relative_position()
            # roi_area = process_image(image=frame.copy(), roi=(x1, y1, x2, y2), device=device)
            scaled_relative_position = motors_relative_position / RELATIVE_POSITION_SCALE
            # tensor_relative_position = torch.tensor(scaled_relative_position, dtype=torch.float32).unsqueeze(0).to(device)
            image = cv2.resize(frame, (200, 66))  # Resize to model input size
            image = image.astype(np.float32) / 255.0  # Normalize
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            # Prepare relative position
            relative_pos = np.array([[scaled_relative_position]], dtype=np.float32)

            # Run inference
            inputs = {"image": image, "relative_position": relative_pos}
            outputs = session.run(None, inputs)

            predicted_x = x1 + (outputs[0][0] * (x2 - x1))

            # Mode decision
            mode_manager.set_current_mode(mode)
            mode, init_flag = mode_manager.decide_next_mode(frame, et)

            # Initialize
            target_x = None
            speed = None

            match mode:
                case Mode.FOLLOW_LEFT_EDGE:
                    target_x, _, mode = action_chain.follow_left_edge(image=frame, predicted_x=predicted_x)
                case Mode.FOLLOW_RIGHT_EDGE:
                    target_x, _, mode = action_chain.follow_right_edge(image=frame, predicted_x=predicted_x)
                case Mode.AVOID_OBSTACLE:
                    _, speed, mode = action_chain.avoid_obstacle(init_flag=init_flag)
                case Mode.CARRY_BOTTLE_PHASE1:
                    target_x, speed, mode = action_chain.carry_bottle_phase1(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE2:
                    target_x, speed, mode = action_chain.carry_bottle_phase2(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE3:
                    target_x, speed, mode = action_chain.carry_bottle_phase3(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE4:
                    target_x, speed, mode = action_chain.carry_bottle_phase4(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE5:
                    target_x, speed, mode = action_chain.carry_bottle_phase5(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE6:
                    target_x, speed, mode = action_chain.carry_bottle_phase6(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE7:
                    target_x, speed, mode = action_chain.carry_bottle_phase7(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE8:
                    target_x, speed, mode = action_chain.carry_bottle_phase8(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE9:
                    target_x, speed, mode = action_chain.carry_bottle_phase9(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE10:
                    target_x, speed, mode = action_chain.carry_bottle_phase10(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE11:
                    target_x, speed, mode = action_chain.carry_bottle_phase11(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE12:
                    target_x, speed, mode = action_chain.carry_bottle_phase12(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE13:
                    target_x, speed, mode = action_chain.carry_bottle_phase13(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE14:
                    target_x, speed, mode = action_chain.carry_bottle_phase14(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE15:
                    target_x, speed, mode = action_chain.carry_bottle_phase15(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE16:
                    target_x, speed, mode = action_chain.carry_bottle_phase16(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE17:
                    target_x, speed, mode = action_chain.carry_bottle_phase17(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE18:
                    target_x, speed, mode = action_chain.carry_bottle_phase18(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.CARRY_BOTTLE_PHASE18:
                    target_x, speed, mode = action_chain.carry_bottle_phase18(
                        image=frame, predicted_x=predicted_x, init_flag=init_flag
                    )
                case Mode.GOAL:
                    break

            if target_x is not None:
                roi_center_x = (x1 + x2) / 2

                offset_pixels = target_x - roi_center_x  # Calculate attitude angle using camera geometry
                theta = calculate_attitude_angle(offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS)
                steering_correction = pid.update(theta)

                left_speed = BASE_SPEED - steering_correction
                right_speed = BASE_SPEED + steering_correction
            elif speed is not None:
                left_speed, right_speed = speed
            else:
                left_speed, right_speed = (0, 0)

            et.set_motor_speed(left_speed=int(left_speed), right_speed=int(right_speed))

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status())  # Send driving information for the real-time inspection

            if send_video_stream and client_socket is not None:
                info: dict[str, Any] = {}
                info["target_x"], info["offset_y"] = target_x, OFFSET_Y

                info["text"] = {
                    "theta_deg": math.degrees(theta),
                    "steering_correction": steering_correction,
                    "left_speed": int(left_speed),
                    "right_speed": int(right_speed),
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

            inf_time = time.time() - inf_start_time
            print(f"Inference time is: {inf_time}")

    finally:
        vs.stop()
        et.stop()

        # Clean up socket connection if it was used
        if send_video_stream and client_socket is not None:
            client_socket.close()

        # Clean up sensor recorder if it was used
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the robot with optional sensor recording and video saving",
        epilog='Example usage: python run.py --model-path "./storage/models/model_left.pt"',
    )
    parser.add_argument("--model-path", required=True, help="Path to the trained model file")
    parser.add_argument("--record-sensor", action="store_true", help="Record sensor data to file")
    parser.add_argument("--save-video", action="store_true", help="Save camera video to file")
    parser.add_argument("--send-video", action="store_true", help="Send video stream to host PC")
    parser.add_argument(
        "--course",
        choices=["left", "right"],
        default="left",
        help="Initial course to follow: 'left' for left edge, 'right' for right edge (default: left)",
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
        course=args.course,
    )
