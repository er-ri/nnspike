#!/usr/bin/env python3
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import cv2
import torch
import random
import pandas as pd
import torchvision.transforms as transforms
from nnspike.utils import draw_driving_info
from nnspike.constant import ROI_CNN
from scripts.utils import process_image
from scripts.utils import load_and_prepare_model

transform = transforms.ToTensor()

x1, y1, x2, y2 = ROI_CNN

FILE_LABEL = "20240827203641_label"
intervals = [1, 1.2, 2, 2.3, 3, 4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


course = "left"  # "left" or "right"
model_paths = [
    f"./storage/models/{course}_interval1_0904_01.pth",
    f"./storage/models/{course}_interval2_0904_01.pth",
    f"./storage/models/{course}_interval3_0909_01.pth",
]


models = [load_and_prepare_model(path, device) for path in model_paths]


def read_label_data(image_path: str = None):
    df = pd.read_csv(f"./storage/frames/{FILE_LABEL}.csv")

    filtered_df = df[
        (df["interval"].isin(intervals))
        & (df["use"] == True)
        & (df["course"] == "left")
    ].copy()
    filtered_df = filtered_df.reset_index(
        drop=True
    )  # Reset index for easier navigation
    image_path = image_path.replace("./", "../") if image_path is not None else None
    index = (
        filtered_df[filtered_df["image_path"] == image_path].index[0]
        if image_path is not None
        else 0
    )

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

        image = cv2.imread(image_path)
        roi_area = process_image(image=image, device=device, roi=ROI_CNN)

        interval = row["interval"]
        if interval == 1.2:
            interval = random.choice([1.0, 2.0])
        elif interval == 2.3:
            interval = random.choice([2.0, 3.0])

        with torch.no_grad():
            if interval == 1:
                output = models[0](roi_area)
            elif interval == 2:
                output = models[1](roi_area)
            elif interval == 3:
                output = models[2](roi_area)

        offset_x = x1 + (output[0][0] * (x2 - x1)).detach().item()
        offset_y = 250

        dir_path, filename = image_path.rsplit("/", 1)

        if not pd.isna(row["adjusted_x"]):
            train_x = row["adjusted_x"]
        elif not pd.isna(row["predicted_x"]):
            train_x = row["predicted_x"]
        else:
            train_x = row["mx"]
        train_y = 250

        info = dict()
        info["offset_x"], info["offset_y"] = offset_x, offset_y
        info["text"] = {
            "image path": filename,
            "offset x": offset_x,
            "difference": offset_x - train_x,
            "type": row["line_type"],
            "interval": interval,
        }
        image = draw_driving_info(image.copy(), info, ROI_CNN)

        image = cv2.circle(
            image.copy(), (int(train_x), train_y), 3, (255, 0, 0), -1
        )  # Training data

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
