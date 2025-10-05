#!/usr/bin/env python3
"""
Camera Angle Calibration Test

This script is used to calibrate and test the attitude angle calculation of the front camera.
It captures video frames, detects line edges using OpenCV, and calculates the attitude angle
based on camera geometry parameters (height and focal length).

The script:
1. Captures frames from the camera
2. Detects line edges at a specified Y offset using get_line_edges_at_y
3. Calculates the attitude angle using camera geometry (height, focal length)
4. Visualizes the detected line centroid
5. Streams the processed video to a host PC for monitoring

This is primarily used for:
- Calibrating camera angle measurements
- Testing the accuracy of attitude angle calculations
- Validating camera geometry parameters
- Visual debugging of line detection

Camera Geometry Parameters:
- CAMERA_HEIGHT: Physical height of the camera from the ground
- CAMERA_FOCAL_LENGTH_PIXELS: Camera focal length in pixels
- ROI_CNN: Region of Interest for processing
- OFFSET_Y: Y-offset for line detection
"""
import os
import sys

# Add parent directory to path to import nnspike modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


import pickle
import socket
import struct
from typing import Optional

import cv2

from nnspike.constants import OFFSET_Y, ROI_CNN
from nnspike.utils import get_line_edges_at_y

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest for OpenCV processing

# Simplified Speed Control Parameters (Easy to tune)
BASE_SPEED = 45  # Base speed for straight lines (adjust this first)

# Socket connection settings
HOST_IP_ADDRESS = "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to

def select_camera_resolution():
    """カメラ解像度を選択する"""
    print("カメラ解像度を選択してください:")
    print("1. 640x480 (VGA) - 標準解像度")
    print("2. 800x600 (SVGA) - 中解像度")
    print("3. 1280x720 (HD 720p) - 高解像度")
    print("4. 1920x1080 (Full HD) - 最高解像度")
    
    while True:
        try:
            choice = input("選択 (1-4): ").strip()
            if choice == "1":
                return 640, 480
            elif choice == "2":
                return 800, 600
            elif choice == "3":
                return 1280, 720
            elif choice == "4":
                return 1920, 1080
            else:
                print("1から4の数字を入力してください。")
        except KeyboardInterrupt:
            print("\nプログラムを終了します。")
            sys.exit(0)
        except:
            print("正しい数字を入力してください。")

# Camera resolution settings
CAMERA_WIDTH, CAMERA_HEIGHT = select_camera_resolution()
print(f"選択された解像度: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


def main():
    # Socket connection for sending camera capture (always enabled)
    client_socket: Optional[socket.socket] = None

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST_IP_ADDRESS, 8485))
        print(f"Connected to host PC at {HOST_IP_ADDRESS}:8485 for video streaming")
    except Exception as e:
        print(f"Warning: Could not connect to host PC for video streaming: {e}")
        if client_socket:
            client_socket.close()
        client_socket = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            left_x, _, _ = get_line_edges_at_y(frame, ROI_CNN, OFFSET_Y, 80)
            target_x = left_x

            if target_x is not None:
                # Calculate position relative to ROI
                mx = target_x - x1  # Relative to ROI
                my = OFFSET_Y - y1  # Relative to ROI

            else:
                # No line detected, use center values
                mx = (x2 - x1) // 2
                my = (y2 - y1) // 2

            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

            # Draw reference lines
            # Yellow horizontal lines at y=300 and y=330
            cv2.line(gray, (0, 300), (CAMERA_WIDTH, 300), (255, 255, 0), 2)
            cv2.line(gray, (0, 330), (CAMERA_WIDTH, 330), (255, 255, 0), 2)
            # Vertical line at center
            cv2.line(gray, (CAMERA_WIDTH // 2, 0), (CAMERA_WIDTH // 2, CAMERA_HEIGHT), (255, 255, 255), 2)

            # Draw centroid point
            cv2.circle(gray, (int(x1 + mx), int(y1 + my)), 5, (255, 255, 255), -1)
            if client_socket is not None:
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
        cap.release()

        # Close socket connection if it was opened
        if client_socket is not None:
            client_socket.close()


if __name__ == "__main__":
    main()
