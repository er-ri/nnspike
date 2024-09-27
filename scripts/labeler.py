#!/usr/bin/env python3
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import cv2
import pandas as pd
from nnspike.utils import draw_driving_info
from nnspike.constant import ROI_CNN


TIMESTAMP = "20240905185809_label"


def read_label_data(image_path: str = None):
    df = pd.read_csv(f"./storage/frames/{TIMESTAMP}.csv")

    filtered_df = df[(df["interval"].isin([1, 1.2, 2, 2.3, 3])) & (df["use"] == True)]
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
        interval = row["interval"]
        image_path = row["image_path"].replace("../", "./")
        image = cv2.imread(image_path)

        if not pd.isna(row["adjusted_x"]):
            offset_x = row["adjusted_x"]
        elif not pd.isna(row["predicted_x"]):
            offset_x = row["predicted_x"]
        else:
            offset_x = row["mx"]
        offset_y = 250  # Constant

        info = dict()
        info["offset_x"], info["offset_y"] = offset_x, offset_y

        dir_path, filename = image_path.rsplit("/", 1)

        info["text"] = {
            "image path": filename,
            "offset x": offset_x,
            "frame": index,
            "line type": row["line_type"],
            "interval": interval,
        }
        image = draw_driving_info(image.copy(), info, ROI_CNN)

        cv2.imshow(f"ETRobot: {dir_path}", image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("u"):  # Update dataframe
            df, index = read_label_data(image_path=image_path)
        elif key == ord("n"):  # Move to next frame
            index += 1
        elif key == ord("b"):  # Move to previous frame
            index -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
