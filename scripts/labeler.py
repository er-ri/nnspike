#!/usr/bin/env python3
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import cv2
import pandas as pd
from nnspike.constants import OFFSET_Y, ROI_CNN, Mode
from nnspike.utils import (
    draw_driving_info,
    find_bottle_center,  # ボトル中心座標・ピクセル数検出
    get_line_edges_at_y,  # 指定Y座標でのライン左右端検出
    get_virtual_line_target_x,  # 仮想ライン左右端検出
    find_blue_target_center,  # 青ターゲット中心座標・ピクセル数検出
    get_is_blue_line_at_y,  # 指定Y座標での青ライン有無判定
    is_x320_on_blue_target,  # 画像中央x=320付近で青ターゲット検出
    is_x320_on_red_target,  # 画像中央x=320付近で赤ターゲット検出
    get_red_target_center_x,  # 赤ターゲット中心x座標取得
    is_left_black_line_detected,  # 左黒ライン検出
    is_horizontal_black_line_detected,  # 水平黒ライン検出
    is_vertical_black_line_detected,  # 垂直黒ライン検出
    is_general_horizontal_line_detected,  # 一般的な水平黒ライン検出
    get_blue_line_pixel,  # 青オブジェクト面積検出
)


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

        offset_y = OFFSET_Y  # Constant value for y-offset in ROI_CNN (new: 350)

        yellow_center_x, _, yellow_pixel_count = find_bottle_center(image, "yellow")
        blue_center_x, _, blue_pixel_count = find_bottle_center(image, "blue")
        red_center_x, _, red_pixel_count = find_bottle_center(image, "red")

        info = dict()
        info["target_x"] = target_x
        info["offset_y"] = offset_y

        dir_path, filename = image_path.rsplit("/", 1)

        motor_a_pos = row["motor_a_relative_position"] if "motor_a_relative_position" in row else "-"
        motor_b_pos = row["motor_b_relative_position"] if "motor_b_relative_position" in row else "-"
        info["text"] = {
            "image path": filename,
            "mode": mode,
            "target_x": int(target_x),
            "yellow": f"center_x={yellow_center_x}, pixel_count={yellow_pixel_count}",
            "blue": f"center_x={blue_center_x}, pixel_count={blue_pixel_count}",
            "red": f"center_x={red_center_x}, pixel_count={red_pixel_count}",
            "frame": row["frame_number"],
            "data type": row["data_type"],
            "right_pos": motor_a_pos,
            "left_pos": motor_b_pos,
        }

        # right_info からは削除
        right_info = []
        right_info.append(f"get_virtual_line_target_x: {get_virtual_line_target_x(image)}")
        right_info.append(f"find_blue_target_center: {find_blue_target_center(image)}")
        right_info.append(f"get_is_blue_line_at_y: {get_is_blue_line_at_y(image)}")
        right_info.append(f"is_x320_on_blue_target: {is_x320_on_blue_target(image)}")
        right_info.append(f"is_x320_on_red_target: {is_x320_on_red_target(image)}")
        right_info.append(f"get_red_target_center_x: {get_red_target_center_x(image)}")
        course_value = row["course"] if "course" in row else "left"
        right_info.append(f"is_left_black_line_detected(course={course_value}): {is_left_black_line_detected(image, course_value)}")
        right_info.append(f"is_horizontal_black_line_detected: {is_horizontal_black_line_detected(image)}")
        right_info.append(f"is_vertical_black_line_detected: {is_vertical_black_line_detected(image)}")
        right_info.append(f"is_general_horizontal_line_detected: {is_general_horizontal_line_detected(image)}")
        right_info.append(f"get_blue_line_pixel: {get_blue_line_pixel(image)}")
        info["right_info"] = right_info

        # y=300の位置に水平線、画面中央に垂直線を描画
        image_with_line = image.copy()
        x_center = image_with_line.shape[1] // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        # 水平線とラベルを複数y座標で描画
        for y_line in [300, 330, 450, 470]:
            # 水平線（黄色）
            cv2.line(image_with_line, (0, y_line), (image_with_line.shape[1], y_line), (0, 255, 255), 2)
            # 水平線の横にy座標値を小さく描画
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

        # 右端にright_infoを表示
        right_x = image_with_line.shape[1] - 10
        start_y = 40
        line_height = 18
        for i, text in enumerate(info["right_info"]):
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            x = right_x - text_size[0]
            y = start_y + i * line_height
            cv2.putText(image_with_line, text, (x, y), font, font_scale, (255, 128, 0), font_thickness, cv2.LINE_AA)

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
