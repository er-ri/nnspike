#!/usr/bin/env python3
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


import cv2
import pandas as pd
from nnspike.constants import OFFSET_Y, ROI_CNN, Mode
from nnspike.utils import draw_driving_info, find_bottle_center, find_blue_target_center


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

        target_w, _, _ = find_blue_target_center(image)
        offset_y = OFFSET_Y  # Constant value for y-offset in ROI_CNN (new: 350)

        _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
        blue_center_x, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
        red_center_x, _, red_pixel_count = find_bottle_center(image=image, color="red")

        info = dict()
        info["target_x"] = target_x
        info["target_w"] = target_w
        info["offset_y"] = offset_y

        dir_path, filename = image_path.rsplit("/", 1)

        info["text"] = {
            "image path": filename,
            "mode": mode,
            "target_x": int(target_x),
            # "target_v": int(target_v),
            "target_w": target_w,
            "yellow_pixel_count": yellow_pixel_count,
            "blue_pixel_count": blue_pixel_count,
            "red_pixel_count": red_pixel_count,
            "blue_center_x": blue_center_x,
            "red_center_x": red_center_x,
            "frame": row["frame_number"],
            "data type": row["data_type"],
        }

        # y=300の位置に水平線、画面中央に垂直線を描画
        image_with_line = image.copy()
        y_line = 300
        x_center = image_with_line.shape[1] // 2
        # 水平線（黄色）
        cv2.line(image_with_line, (0, y_line), (image_with_line.shape[1], y_line), (0, 255, 255), 2)
        # 水平線の横にy座標値を小さく描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_y = f"y={y_line}"
        text_y_size, _ = cv2.getTextSize(text_y, font, font_scale, font_thickness)
        text_y_x = 5
        text_y_y = y_line - 7 if y_line - 7 > text_y_size[1] else y_line + text_y_size[1] + 7
        cv2.putText(image_with_line, text_y, (text_y_x, text_y_y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        # 垂直線（黄色）
        cv2.line(image_with_line, (x_center, 0), (x_center, image_with_line.shape[0]), (0, 255, 255), 2)
        # 垂直線の横にx座標値を小さく描画
        text_x_label = f"x={x_center}"
        text_x_size, _ = cv2.getTextSize(text_x_label, font, font_scale, font_thickness)
        text_x_x = x_center + 7 if x_center + 7 + text_x_size[0] < image_with_line.shape[1] else x_center - text_x_size[0] - 7
        text_x_y = 20
        cv2.putText(image_with_line, text_x_label, (text_x_x, text_x_y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        
        image = draw_driving_info(image_with_line, info, ROI_CNN)

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
