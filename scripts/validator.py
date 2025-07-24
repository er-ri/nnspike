#!/usr/bin/env python3
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


from typing import Optional

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms

from nnspike.constants import OFFSET_Y, RELATIVE_POSITION_SCALE, ROI_CNN
<<<<<<< HEAD
from nnspike.models import NvidiaModel
from nnspike.utils import draw_driving_info
from scripts.utils import process_image
=======
from nnspike.utils import draw_driving_info
from scripts.utils import load_optimized_model, process_image
>>>>>>> wip/teamwork-li

transform = transforms.ToTensor()

x1, y1, x2, y2 = ROI_CNN
<<<<<<< HEAD

FILE_LABEL = "20250705153842_label"
=======
>>>>>>> wip/teamwork-li

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


<<<<<<< HEAD
course = "left"  # "left" or "right"


model = NvidiaModel()
model.load_state_dict(torch.load("./storage/models/model_left_0713.pth", map_location=device))
model.eval()


def read_label_data(image_path: Optional[str] = None):
    df = pd.read_csv(f"./storage/labels/{FILE_LABEL}.csv")

    filtered_df = df[(df["use"] == True) & (df["course"] == "left")].copy()
=======
def read_label_data(csv_file: str, image_path: Optional[str] = None):
    df = pd.read_csv(csv_file)

    filtered_df = df[(df["use"] == True)].copy()
>>>>>>> wip/teamwork-li
    filtered_df = filtered_df.reset_index(drop=True)  # Reset index for easier navigation
    index = filtered_df[filtered_df["image_path"] == image_path].index[0] if image_path is not None else 0

    return filtered_df, index


def main():
    parser = argparse.ArgumentParser(
        description="Validate neural network model predictions against labeled training data",
        epilog="""
Examples:
  %(prog)s --model-path ./storage/models/model.pt --label-data ./storage/labels/data.csv

Controls:
  q - Quit the application
  n - Move to next frame
  b - Move to previous frame
  u - Update/reload label data for current image

The validator displays model predictions overlaid on images with:
  - Red circle: Training data target position
  - Green overlay: Model prediction visualization
  - Text info: Prediction details, differences, and probabilities
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to the trained PyTorch model file (.pth)")
    parser.add_argument(
        "--label-data", required=True, help="Path to the CSV file containing labeled training data with columns: image_path, target_x, motor_a_relative_position, use, data_type"
    )
    args = parser.parse_args()

    model = load_optimized_model(args.model_path, device=device)

    df, _ = read_label_data(args.label_data)

    index = 0

    while True:
        if index < 0:
            index = 0
        elif index >= len(df):
            index = len(df) - 1

<<<<<<< HEAD
=======
        print(f"Processing index: {index}/{len(df) - 1}")
>>>>>>> wip/teamwork-li
        row = df.iloc[index]
        image_path = row["image_path"].replace("../", "./")
        relative_position = abs(row["motor_a_relative_position"] / RELATIVE_POSITION_SCALE)

        image = cv2.imread(image_path)
        roi_area = process_image(image=image, device=device, roi=(x1, y1, x2, y2))
        relative_position = torch.tensor(relative_position, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(roi_area, relative_position)

        prob, mode = torch.max(outputs[0], dim=1)

<<<<<<< HEAD
        offset_x = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()
=======
        target_x = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()
>>>>>>> wip/teamwork-li
        offset_y = OFFSET_Y

        dir_path, filename = image_path.rsplit("/", 1)

        train_x = row["target_x"]
        train_y = OFFSET_Y

        info = dict()
        info["target_x"], info["offset_y"] = target_x, offset_y
        info["text"] = {
            "image path": filename,
<<<<<<< HEAD
            "offset x": offset_x,
            "difference": offset_x - train_x,
=======
            "offset x": target_x,
            "difference": target_x - train_x,
>>>>>>> wip/teamwork-li
            "mode": mode.item(),
            "probability": round(prob[0].item(), 2),
            "type": row["data_type"],
        }
        image = draw_driving_info(image.copy(), info, (x1, y1, x2, y2))

        image = cv2.circle(image.copy(), (int(train_x), train_y), 3, (255, 0, 0), -1)  # Training data

        cv2.imshow(f"ETRobot: {dir_path}", image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("u"):
            df, index = read_label_data(args.label_data, image_path=image_path)
        elif key == ord("n"):  # Move to next frame
            index += 1
        elif key == ord("b"):  # Move to previous frame
            index -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
