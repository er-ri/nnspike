#!/usr/bin/env python3
import sys
import cv2
import time
import torch
import select
from nnspike.unit import ETRobot
from nnspike.utils import PIDController
from nnspike.constant import ROI_CNN
from scripts.utils import load_and_prepare_model, process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest
LOW_POWER = 35
HIGH_POWER = 50


course = "left"  # "left" or "right"
model_paths = [
    f"./storage/models/{course}_interval1_0904_01.pth",
    f"./storage/models/{course}_interval2_0904_01.pth",
    f"./storage/models/{course}_interval3_0909_01.pth",
]


models = [load_and_prepare_model(path, device) for path in model_paths]


# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


def main():
    # Initialization
    et = ETRobot()
    pid = PIDController(
        Kp=35, Ki=0, Kd=0, setpoint=0, output_limits=(-base_power, base_power)
    )
    time.sleep(0.5)

    et.reset_motor()

    base_power = HIGH_POWER
    interval_idx = 0

    print("Get started by waving your hand in front of the ultrasound sensor..")
    while True:
        if et.ultrasonic_sensor is not None and et.ultrasonic_sensor < 10:
            break

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        roi_area = process_image(image=frame.copy(), device=device, roi=ROI_CNN)

        # Decide interval
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

        base_power = LOW_POWER if abs(distance) > 0.5 else HIGH_POWER
        pid.set_output_limits((-base_power, base_power))
        # Steering angle range: -1 ~ 1 (1: turn right; -1: turn left)
        steer = pid.update(distance)

        left_power = base_power - steer
        right_power = base_power + steer

        et.set_motor_power(left_power=int(left_power), right_power=int(right_power))
        print(f"Distance: {distance}; Motor Counter: {motor_count};")

        # User input to exit the running loop
        i, o, e = select.select([sys.stdin], [], [], 0)
        if i:
            input_char = sys.stdin.read(1)
            if input_char == "b":
                print("Exiting...")
                break

    et.stop()
    cap.release()


if __name__ == "__main__":

    main()
