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
        )

    # Socket connection for sending camera capture
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
    )
    
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
            )            # Dynamic speed control using external function
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
                
            # Get spike status once for efficiency and consistency
            spike_status = et.get_spike_status()
            
            # Color sensor formatting
            def format_color_sensor(status):
                if status and status.sensors and status.sensors.color:
                    color = status.sensors.color
                    return f"R:{color.reflected} A:{color.ambient} C:{color.color}"
                else:
                    return "R:N/A A:N/A C:N/A"
            
            # Ultrasonic sensor formatting
            def format_ultrasonic_sensor(status):
                if status and status.sensors and status.sensors.distance is not None:
                    return f"{status.sensors.distance}cm"
                else:
                    return "N/A cm"
            
            # Prepare driving information for visualization
            info = dict()
            info["offset_x"], info["offset_y"] = x1 + mx, y1 + my
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
                "color_sensor": format_color_sensor(spike_status),
                "ultrasonic_sensor": format_ultrasonic_sensor(spike_status),
                "contour_area": f"{int(cv2.contourArea(max_contour)) if max_contour is not None else 0}px2",
            }

            # Create visualization frame
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray = draw_driving_info(gray, info, (x1, y1, x2, y2))

            # Draw contour on the visualization if found
            if max_contour is not None:
                # Adjust contour coordinates to full frame
                adjusted_contour = max_contour + np.array([x1, y1])
                cv2.drawContours(gray, [adjusted_contour], -1, (255, 255, 255), 2)

                # Draw centroid
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
    except Exception as e:
        print(f"Error: {e}")
    finally:
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
