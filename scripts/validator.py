#!/usr/bin/env python3
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
from nnspike.models import NvidiaModel
from nnspike.utils import draw_driving_info
from scripts.utils import process_image

transform = transforms.ToTensor()

x1, y1, x2, y2 = ROI_CNN

FILE_LABEL = "20250705153842_label"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


course = "left"  # "left" or "right"


model = NvidiaModel()
model.load_state_dict(torch.load("./storage/models/model_left_0713.pth", map_location=device))
model.eval()


def read_label_data(image_path: Optional[str] = None):
    df = pd.read_csv(f"./storage/labels/{FILE_LABEL}.csv")

    filtered_df = df[(df["use"] == True) & (df["course"] == "left")].copy()
    filtered_df = filtered_df.reset_index(drop=True)  # Reset index for easier navigation
    index = filtered_df[filtered_df["image_path"] == image_path].index[0] if image_path is not None else 0

    return filtered_df, index


def main():
    df, _ = read_label_data()

    index = 0
    while True:
        if index < 0:
            index = 0
        elif index >= len(df):
            index = len(df) - 1

        row = df.iloc[index]
        image_path = row["image_path"].replace("../", "./")
        relative_position = abs(row["motor_a_relative_position"] / RELATIVE_POSITION_SCALE)

        image = cv2.imread(image_path)
        roi_area = process_image(image=image, device=device, roi=(x1, y1, x2, y2))
        relative_position = torch.tensor(relative_position, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(roi_area, relative_position)

        prob, mode = torch.max(outputs[0], dim=1)

        offset_x = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()
        offset_y = OFFSET_Y

        dir_path, filename = image_path.rsplit("/", 1)

        train_x = row["target_x"]
        train_y = OFFSET_Y

        info = dict()
        info["offset_x"], info["offset_y"] = offset_x, offset_y
        info["text"] = {
            "image path": filename,
            "offset x": offset_x,
            "difference": offset_x - train_x,
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
            df, index = read_label_data(image_path=image_path)
        elif key == ord("n"):  # Move to next frame
            index += 1
        elif key == ord("b"):  # Move to previous frame
            index -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
