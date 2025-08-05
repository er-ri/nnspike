#!/usr/bin/env python3
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import cv2
import pandas as pd

from nnspike.constants import OFFSET_Y, ROI_CNN, Mode
from nnspike.utils import draw_driving_info, find_bottle_center


def read_label_data(label_path: str, image_path: str | None = None):
    df = pd.read_csv(label_path)

    filtered_df = df[df["use"] == True]
    filtered_df = filtered_df.reset_index(drop=True)  # Reset index for easier navigation
    image_path = image_path.replace("./", "../") if image_path is not None else None
    index = filtered_df[filtered_df["image_path"] == image_path].index[0] if image_path is not None else 0

    return filtered_df, index


def main():
    parser = argparse.ArgumentParser(
        description="Label data viewer for ETRobot",
        epilog='Example: python scripts/labeler.py --label-data "./storage/labels/timestamp_label.csv"',
    )
    parser.add_argument(
        "--label-data",
        required=True,
        help="Path to the label CSV file.",
    )
    args = parser.parse_args()

    label_path = args.label_data
    df, _ = read_label_data(label_path)

    index = 0
    while True:
        if index < 0:
            index = 0
        elif index >= len(df):
            index = len(df) - 1

        row = df.iloc[index]
        mode = row["mode"]
        target_x = row["target_x"] if not pd.isna(row["target_x"]) else 0
        # interval = row["interval"]
        image_path = row["image_path"].replace("../", "./")
        image = cv2.imread(image_path)
        from nnspike.utils import get_virtual_line_edges_at_y, find_blue_target_center
        target_v = get_virtual_line_edges_at_y(image, OFFSET_Y)
        target_w, _, _ = find_blue_target_center(image)

        offset_y = OFFSET_Y  # Constant value for y-offset in ROI_CNN (new: 350)

        _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
        blue_center_x, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
        red_center_x, _, red_pixel_count = find_bottle_center(image=image, color="red")

        info = dict()
        info["target_x"] = target_x
        info["target_v"] = target_v
        info["target_w"] = target_w
        info["offset_y"] = offset_y

        dir_path, filename = image_path.rsplit("/", 1)

        info["text"] = {
            "image path": filename,
            "mode": mode,
            "target_x": int(target_x),
            "target_v": int(target_v),
            "target_w": target_w,
            "yellow_pixel_count": yellow_pixel_count,
            "blue_pixel_count": blue_pixel_count,
            "red_pixel_count": red_pixel_count,
            "blue_center_x": blue_center_x,
            "red_center_x": red_center_x,
            "frame": row["frame_number"],
            "data type": row["data_type"],
        }
        image = draw_driving_info(image.copy(), info, ROI_CNN)

        cv2.imshow(f"ETRobot: {dir_path}", image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):  # Update dataframe
            df, index = read_label_data(label_path, image_path=image_path)
            index = 578
        elif key == ord("2"):  # Update dataframe
            df, index = read_label_data(label_path, image_path=image_path)
            index = 734
        elif key == ord("3"):  # Update dataframe
            df, index = read_label_data(label_path, image_path=image_path)
            index = 904
        elif key == ord("4"):  # Update dataframe
            df, index = read_label_data(label_path, image_path=image_path)
            index = 1009
        elif key == ord("u"):  # Update dataframe
            df, index = read_label_data(label_path, image_path=image_path)
        elif key == ord("n"):  # Move to next frame
            index += 1
        elif key == ord("b"):  # Move to previous frame
            index -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
