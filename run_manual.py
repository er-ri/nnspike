#!/usr/bin/env python3
"""
OpenCV-Based Line Following Robot Control

This script controls a line-following robot using OpenCV for image processing
instead of neural network predictions. It uses the get_line_edges_at_y function
to detect the line centroid and follows it using PID control.

Speed Tuning Parameters:
- BASE_SPEED: Base speed for straight lines (start here)

PID Tuning Parameters:
- Kp, Ki, Kd: Standard PID parameters for steering correction
  * Kp (Proportional): Controls immediate response to error
    - Too high: Causes zigzag/oscillation
    - Too low: Slow response, may not follow turns
    - Start with: 10-20 for line following
  * Ki (Integral): Eliminates steady-state error
    - Too high: Causes instability and overshoot
    - Too low: Robot may drift to one side
    - Start with: 0.1-1.0
  * Kd (Derivative): Smooths out rapid changes
    - Too high: Sensitive to noise, erratic behavior
    - Too low: May overshoot on turns
    - Start with: 2-10
"""
import cv2
import math
import time
import socket
import pickle
import struct
import argparse
import numpy as np
import sys
from nnspike.unit import ETRobot
from nnspike.unit.actions import avoid_obstacle, catch_bottle_blue
from nnspike.utils.control import find_bottle_center, find_bottle_center_with_yellow_count, find_bottle_center_with_blue_count, find_gate_center

# Platform-specific imports for keyboard input
try:
    import msvcrt  # Windows

    WINDOWS = True
except ImportError:
    import select
    import tty
    import termios

    WINDOWS = False
from nnspike.utils import (
    get_line_edges_at_y,
    draw_driving_info,
    PIDController,
    SensorRecorder,
    calculate_attitude_angle,
)
from nnspike.constants import (
    ROI_CNN,
    OFFSET_Y,
    Mode,
    CAMERA_HEIGHT,
    CAMERA_FOCAL_LENGTH_PIXELS,
)

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)

# Socket connection settings
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


class KeyboardController:
    def __init__(self):
        self.running = True
        self.current_key = None

        if not WINDOWS:
            # Save terminal settings for Unix-like systems
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

    def get_key(self):
        """Get a single keypress"""
        if WINDOWS:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8").lower()
                return key
        else:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1).lower()
                return key
        return None

    def cleanup(self):
        """Restore terminal settings"""
        if not WINDOWS:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main(record_sensor_data=False, save_camera_video=False, send_video_stream=False):
    # Initialize edge following preference
    mode = Mode.LEFT_EDGE_FOLLOWING  # 0 for left edge, 1 for right edge
    previous_mode = Mode.LEFT_EDGE_FOLLOWING  # Initialize previous_mode

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
        sensor_recorder.start_recording()  # Initialize video writer conditionally
    video_writer = None
    if save_camera_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
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
            client_socket = None

    # Initialize robot, PID controller, and keyboard controller
    et = ETRobot()
    keyboard = KeyboardController()
    pid = PIDController(
        Kp=50,  # Reduced from 50 to minimize zigzag behavior
        Ki=0,  # Small integral term to eliminate steady-state error
        Kd=5,  # Derivative term to smooth out rapid changes
        setpoint=0,
        output_limits=(
            -BASE_SPEED,
            BASE_SPEED,
        ),  # Direct radian limits for steering correction
    )

    et.move_arm(0, 1.0)
    et.move_arm(2, 0.5)
    et.set_motor_relative_position(left_positon=0, right_position=0)

    frame_count = 0  # Initialize frame counter

    try:
        while et.is_running and keyboard.running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_count += 1  # Increment frame counter

            # Save video frame if enabled
            if save_camera_video and video_writer is not None:
                video_writer.write(frame)

            # Check for keyboard input to change behavior mode
            key = keyboard.get_key()
            if key == "q":  # 'q' key to quit
                print("Quitting...")
                keyboard.running = False
                break
            elif key == "a":  # 'a' key for left
                mode = Mode.LEFT_EDGE_FOLLOWING
                print("Switched to following: left edge")
            elif key == "d":  # 'd' key for right
                mode = Mode.RIGHT_EDGE_FOLLOWING
                print("Switched to following: right edge")
            elif key == "c":  # 'c' key for bottle carrying
                mode = Mode.BOTTLE_CARRYING
                print("Switched to bottle carrying mode")
            elif key == "b":  # 'b' key for blue bottle catching
                mode = Mode.BOTTLE_CATCH_BLUE
                print("Switched to blue bottle catching mode")
            elif key == "g":  # 'g' key for blue bottle to gate carrying
                mode = Mode.BLUE_BOTTLE_TO_GATE
                print("Switched to blue bottle to gate carrying mode")
            elif key == "o":  # 'o' key to avoid obstacle
                previous_mode = mode  # Save current mode
                mode = Mode.OBSTACLE_AVOIDANCE
                # avoid_obstacle(et)  # Avoid obstacle with a turn
                print("Avoiding obstacle...")
                # Reset counters when entering obstacle avoidance mode
                if hasattr(main, 'no_obstacle_counter'):
                    main.no_obstacle_counter = 0
                if hasattr(main, 'error_counter'):
                    main.error_counter = 0

            match mode:
                case Mode.LEFT_EDGE_FOLLOWING:
                    left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = left_x
                case Mode.RIGHT_EDGE_FOLLOWING:
                    _, right_x, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
                    target_x = right_x
                case Mode.BOTTLE_CARRYING:
                    (cx, _), _ = find_bottle_center(frame)
                    target_x = cx
                case Mode.OBSTACLE_AVOIDANCE:
                    # Check if previous_mode is defined, if not set default
                    if 'previous_mode' not in locals():
                        previous_mode = Mode.LEFT_EDGE_FOLLOWING
                    
                    # In obstacle avoidance mode, use yellow bottle detection
                    try:
                        yellow_result = find_bottle_center_with_yellow_count(frame)
                        
                        if frame_count % 30 == 0:  # Print debug info every 30 frames
                            print(f"DEBUG: Yellow detection result: {yellow_result}")
                        
                        if yellow_result[0] is not None:
                            (cx, _), _, yellow_pixel_count = yellow_result
                            target_x = cx
                            
                            if frame_count % 30 == 0:
                                print(f"DEBUG: Yellow bottle detected - center: ({cx}, _), pixel count: {yellow_pixel_count}")
                            
                            # If yellow pixel count exceeds threshold, execute obstacle avoidance
                            if yellow_pixel_count > 8000:  # Lower threshold for better detection
                                if frame_count % 30 == 0:
                                    print("DEBUG: Yellow pixel count exceeds threshold, executing obstacle avoidance")
                                avoid_obstacle(et)
                                print("Avoiding obstacle...")
                                
                                # After avoiding obstacle, return to previous mode
                                mode = previous_mode
                                print(f"Obstacle avoided, returning to {previous_mode.name} mode")
                                
                                # Reset no-obstacle counter
                                if hasattr(main, 'no_obstacle_counter'):
                                    main.no_obstacle_counter = 0
                            else:
                                if frame_count % 30 == 0:
                                    print(f"DEBUG: Yellow pixel count ({yellow_pixel_count}) below threshold (8000)")
                                
                                # If no obstacle detected for several frames, return to previous mode
                                if not hasattr(main, 'no_obstacle_counter'):
                                    main.no_obstacle_counter = 0
                                main.no_obstacle_counter += 1
                                
                                if main.no_obstacle_counter > 60:  # After 60 frames (~2 seconds) with no obstacle
                                    mode = previous_mode
                                    print(f"No obstacle detected for 60 frames, returning to {previous_mode.name} mode")
                                    main.no_obstacle_counter = 0
                        else:
                            if frame_count % 30 == 0:
                                print("DEBUG: No yellow bottle detected")
                            target_x = None
                            
                            # If no yellow bottle detected for several frames, return to previous mode
                            if not hasattr(main, 'no_obstacle_counter'):
                                main.no_obstacle_counter = 0
                            main.no_obstacle_counter += 1
                            
                            if main.no_obstacle_counter > 90:  # After 90 frames (~3 seconds) with no detection
                                mode = previous_mode
                                print(f"No yellow bottle detected for 90 frames, returning to {previous_mode.name} mode")
                                main.no_obstacle_counter = 0
                            
                    except Exception as e:
                        print(f"DEBUG: Error in OBSTACLE_AVOIDANCE mode: {e}")
                        target_x = None
                        # On error, return to previous mode after a delay
                        if not hasattr(main, 'error_counter'):
                            main.error_counter = 0
                        main.error_counter += 1
                        
                        if main.error_counter > 30:  # After 30 frames with errors
                            mode = previous_mode
                            print(f"Persistent errors in obstacle avoidance, returning to {previous_mode.name} mode")
                            main.error_counter = 0
                case Mode.BOTTLE_CATCH_BLUE:
                    # In blue bottle catching mode, use blue bottle detection
                    (cx, _), _, blue_pixel_count = find_bottle_center_with_blue_count(frame)
                    target_x = cx
                    
                    # If blue pixel count exceeds thresholqd, execute blue bottle catching
                    if blue_pixel_count > 15000:
                        catch_bottle_blue(et)
                        print("Blue bottle caught!")
                case Mode.BLUE_BOTTLE_TO_GATE:
                    # In blue bottle to gate carrying mode, use gate detection
                    gate_result = find_gate_center(frame)
                    if gate_result[0] is not None:
                        (cx, _), confidence = gate_result
                        target_x = cx
                        print(f"Gate detected at ({cx}, _), confidence: {confidence:.3f}")
                    else:
                        # If no gate detected, head to center
                        target_x = (x1 + x2) // 2  # Screen center
                        print("Gate not detected, heading to center")
                case _:
                    # Default to center if invalid edge specified
                    target_x = (left_x + right_x) // 2

            if target_x is not None:
                # Calculate position relative to ROI
                mx = target_x - x1  # Relative to ROI
                my = OFFSET_Y - y1  # Relative to ROI

                # Calculate offset from ROI center
                roi_center_x = (x2 - x1) // 2
                offset_pixels = mx - roi_center_x

                # Create a simple contour for visualization (approximate target point)
                max_contour = np.array([[[mx, my]]], dtype=np.int32)
            else:
                # No line detected, use center values
                mx = (x2 - x1) // 2
                my = (y2 - y1) // 2
                offset_pixels = 0
                max_contour = None  # Calculate attitude angle using camera geometry

            theta = calculate_attitude_angle(
                offset_pixels, OFFSET_Y, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELS
            )  # Use simplified speed control
            current_base_speed = BASE_SPEED

            steering_correction = pid.update(theta)

            # Apply simple differential steering
            left_speed = current_base_speed - steering_correction
            right_speed = current_base_speed + steering_correction

            # Clamp speed values to valid range
            left_speed = int(max(0, min(100, left_speed)))
            right_speed = int(max(0, min(100, right_speed)))

            # DEBUG: Print motor speeds only when mode changes or occasionally
            if frame_count % 30 == 0:  # Print every 30 frames (~1 second at 30fps)
                print(f"DEBUG: Motor speeds - Left: {left_speed}, Right: {right_speed}, Mode: {mode.name}")

            # et.set_motor_forward_speed(
            #     left_speed=left_speed,
            #     right_speed=right_speed,
            # )

            # Log sensor data using the recorder if enabled
            if record_sensor_data and sensor_recorder is not None:
                sensor_recorder.log_frame_data(et.get_spike_status(), mode)

            status = et.get_spike_status()
            left_pos = status.motors["A"].relative_position
            right_pos = status.motors["B"].relative_position

            info = dict()
            info["offset_x"], info["offset_y"] = x1 + mx, y1 + my
            info["text"] = {
                "mode": mode.name,
                "left_relative_position": left_pos,
                "right_relative_position": right_pos,
                "theta_deg": round(math.degrees(theta), 2),
                "steering_correction": round(steering_correction, 2),
                "left_speed": left_speed,
                "right_speed": right_speed,
            }

            # Create visualization frame
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray = draw_driving_info(gray, info, (x1, y1, x2, y2))

            # Draw contour on the visualization if found
            if max_contour is not None:
                # Adjust contour coordinates to full frame
                adjusted_contour = max_contour + np.array([x1, y1])
                cv2.drawContours(
                    gray, [adjusted_contour], -1, (255, 255, 255), 2
                )  # Draw centroid
                cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)
            if send_video_stream and client_socket is not None:
                try:
                    ret, buffer = cv2.imencode(".jpg", gray)
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
        et.stop()
        cap.release()

        # Close socket connection if it was opened
        if client_socket is not None:
            client_socket.close()

        # Clean up video writer if it was used
        if save_camera_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")

        # Clean up sensor recorder if it was used
        if record_sensor_data and sensor_recorder is not None:
            sensor_recorder.stop_recording()
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")
        # Restore terminal settings on exit
        keyboard.cleanup()


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
    parser.add_argument(
        "--send-video", action="store_true", help="Send video stream to host PC"
    )

    args = parser.parse_args()

    print("Starting OpenCV-based line following robot...")
    print(f"Using ROI: {ROI_CNN}")
    print(f"Base speed: {BASE_SPEED}")
    print("Default: Following left edge")
    print(f"Video streaming to host PC: {'Enabled' if args.send_video else 'Disabled'}")
    print("Controls:")
    print("  'a' - Follow left edge")
    print("  'd' - Follow right edge")
    print("  'c' - Bottle carrying mode")
    print("  'b' - Blue bottle catching mode")
    print("  'g' - Blue bottle to gate carrying mode")
    print("  'o' - Obstacle avoidance")
    print("  'q' - Quit")
    print("Press Ctrl+C to stop")

    main(
        record_sensor_data=args.record_sensor,
        save_camera_video=args.save_video,
        send_video_stream=args.send_video,
    )
