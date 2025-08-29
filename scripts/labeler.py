#!/usr/bin/env python3
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import cv2
import pandas as pd

from nnspike.constants import OFFSET_Y, ROI_CNN
from nnspike.utils import draw_driving_info


def read_label_data(label_path: str, image_path: str | None = None):
    df = pd.read_csv(label_path)

    image_path = image_path.replace("./", "../") if image_path is not None else None
    index = df[df["image_path"] == image_path].index[0] if image_path is not None else 0

    return df, index


# Define the mouse callback function
def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        df = param["df"]
        row = param["row"]

        df.at[row.name, "target_x"] = x
        print(f"Target x set at: ({x})")

    if event == cv2.EVENT_RBUTTONDOWN:
        df = param["df"]
        row = param["row"]
        index = param["index"]

        df.at[row.name, "use"] = False
        print(f"Frame {index} marked as unused.")


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

        image_path = row["image_path"].replace("../", "./")
        image = cv2.imread(image_path)
        clone_image = image.copy()

        info = dict()
        info["target_x"], info["offset_y"] = target_x, OFFSET_Y

        dir_path, filename = image_path.rsplit("/", 1)

        info["text"] = {
            "image path": filename,
            "mode": mode,
            "use": row["use"],
            "target_x": target_x,
        }
        clone_image = draw_driving_info(clone_image, info, ROI_CNN)

        # Darken the image if not in use
        clone_image = cv2.convertScaleAbs(clone_image, alpha=1.0, beta=-100) if row["use"] == False else clone_image

        clone_image = (
            cv2.putText(clone_image, "No Target", (240, 320), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 0), 2, cv2.LINE_4)
            if (row["use"] == True) and (pd.isna(row["target_x"]))
            else clone_image
        )

        cv2.imshow(f"ETRobot: {dir_path}", clone_image)

        frame_record = {"df": df, "row": row, "index": index}
        cv2.setMouseCallback(f"ETRobot: {dir_path}", mouse_callback, frame_record)  # type: ignore

        key = cv2.waitKey(1) & 0xFF  # Wait indefinitely for a key press
        if key == ord("q"):
            break
        elif key == ord("f"):
            df.at[row.name, "use"] = not row["use"]
        elif key == ord("u"):  # Update dataframe
            df, index = read_label_data(label_path, image_path=image_path)
            print(f"Updated dataframe from {label_path}")
        elif key == ord("s"):
            try:
                df.to_csv(label_path, index=False)
                print(f"Saved labels to {label_path}")
            except PermissionError as e:
                print(e)
        elif key == ord("n"):  # Move to next frame
            index += 1
        elif key == ord("b"):  # Move to previous frame
            index -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
