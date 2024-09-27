#!/usr/bin/env python3
import cv2
import time
import socket
import pickle
import struct
import torch
from nnspike.unit import ETRobot
from nnspike.utils import draw_driving_info, PIDController
from nnspike.constant import ROI_CNN
from scripts.utils import load_and_prepare_model, process_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# User defined constants
TIMESTAMP = time.strftime(
    "%Y%m%d%H%M%S", time.localtime()
)  # File Label for saving camera capture and steering data
x1, y1, x2, y2 = ROI_CNN  # Region of Interest
LOW_POWER = 35
HIGH_POWER = 50
HOST_IP_ADDRESS = (
    "192.168.137.1"  # The destination IP(PC) that the Raspberry Pi will send to
)

course = "left"  # "left" or "right"
model_paths = [
    f"./storage/models/{course}_interval1_0904_01.pth",
    f"./storage/models/{course}_interval2_0904_01.pth",
    f"./storage/models/{course}_interval3_0909_01.pth",
]

models = [load_and_prepare_model(path, device) for path in model_paths]

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    filename=f"storage/videos/{TIMESTAMP}_picamera.avi",
    fourcc=fourcc,
    fps=30,
    frameSize=(640, 480),
)

# Socket connection for sending camera capture
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST_IP_ADDRESS, 8485))


def main():
    # Initialization
    et = ETRobot()
    pid = PIDController(
        Kp=25, Ki=0, Kd=0, setpoint=0, output_limits=(-base_power, base_power)
    )

    time.sleep(0.5)

    et.reset_motor()
    interval_idx = 0
    base_power = HIGH_POWER

    while et.is_running == True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        out.write(frame)

        roi_area = process_image(image=frame.copy(), device=device, roi=ROI_CNN)

        # Todo: stage determination
        motor_count = et.motor_count if et.motor_count is not None else 0
        if motor_count >= 0 and motor_count < 32:
            interval_idx = 0
        elif motor_count >= 32 and motor_count < 62:
            interval_idx = 1
        elif motor_count >= 62 and motor_count < 1000:
            interval_idx = 2

        with torch.no_grad():
            output = models[interval_idx](roi_area)

        centroid_x = (x2 - x1) / 2
        offset_x = (x1 + output * (x2 - x1))[0][0].detach().item()
        distance = (offset_x - x1 - centroid_x) / centroid_x

        # Steering angle range: -1 ~ 1 (1: trun right; -1: turn left)
        base_power = LOW_POWER if abs(distance) > 0.5 else HIGH_POWER
        pid.set_output_limits(-base_power, base_power)
        steer = pid.update(distance)

        left_power = base_power - steer
        right_power = base_power + steer

        et.set_motor_power(left_power=int(left_power), right_power=int(right_power))

        # Send driving information for the real-time inspection
        info = dict()
        info["offset_x"], info["offset_y"] = offset_x, 250
        info["text"] = {
            "distance": distance,
            "steer": steer,
            "motor count": motor_count,
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

    et.stop()
    cap.release()
    out.release()


if __name__ == "__main__":
    main()
