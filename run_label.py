
import os
import cv2
import pandas as pd
from nnspike.utils import (
    find_bottle_center,
    get_virtual_line_target_x,
    find_blue_target_center,
    is_x320_on_blue_target,
    is_x320_on_red_target,
    get_red_target_center_x,
    is_left_black_line_detected,
    is_general_horizontal_line_detected,
    is_horizontal_black_line_detected,
    is_vertical_black_line_detected,
    get_is_blue_line_at_y,
    get_blue_line_pixel,
)
from nnspike.constants import ROI_CNN


VIDEO_DIR = r"C:\Users\MSAD\github\nnspike\storage\20250824\videos"
LABEL_DIR = r"C:\Users\MSAD\github\nnspike\storage\20250824\labels_plus"
os.makedirs(LABEL_DIR, exist_ok=True)

x1, y1, x2, y2 = ROI_CNN

def label_video_and_sensor(video_path, label_path):
    import csv
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    label_rows = []
    # 既存label.csvからP列（コース情報）を取得
    with open(label_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        label_csv_rows = list(reader)
    # P列は16番目（0-indexで15）と仮定
    while cap.isOpened():
        ret, image = cap.read()
        if not ret or frame_idx >= len(label_csv_rows)-1:
            break
        # ラベル情報のみ生成
        label = {}
        yellow_center_x, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
        blue_center_x, _, blue_pixel_count = find_bottle_center(image=image, color="blue")
        red_center_x, _, red_pixel_count = find_bottle_center(image=image, color="red")
        label["yellow_center_x"] = yellow_center_x
        label["yellow_pixel_count"] = yellow_pixel_count
        label["blue_center_x"] = blue_center_x
        label["blue_pixel_count"] = blue_pixel_count
        label["red_center_x"] = red_center_x
        label["red_pixel_count"] = red_pixel_count
        label["get_virtual_line_target_x"] = get_virtual_line_target_x(image)
        label["find_blue_target_center"] = find_blue_target_center(image)
        label["is_x320_on_blue_target"] = is_x320_on_blue_target(image)
        label["is_x320_on_red_target"] = is_x320_on_red_target(image)
        label["get_red_target_center_x"] = get_red_target_center_x(image)
        # P列（コース情報）を取得
        course_value = label_csv_rows[frame_idx+1][15] if len(label_csv_rows[frame_idx+1]) > 15 else "left"
        label["is_left_black_line_detected"] = is_left_black_line_detected(image, course_value)
        label["is_general_horizontal_line_detected"] = is_general_horizontal_line_detected(image)
        label["is_horizontal_black_line_detected"] = is_horizontal_black_line_detected(image, intersection_y=450, roi=(250, 300, 390, 540))
        label["is_vertical_black_line_detected"] = is_vertical_black_line_detected(image, roi=(100, 200, 540, 540), center_tolerance=120)
        label["get_is_blue_line_at_y"] = get_is_blue_line_at_y(image)
        label["get_blue_line_pixel"] = get_blue_line_pixel(image)
        label_rows.append(label)
        frame_idx += 1
    cap.release()
    label_df = pd.DataFrame(label_rows)
    label_df.index.name = "frame_idx"
    # 既存label.csvとラベル情報をR列以降に追加
    orig_df = pd.read_csv(label_path)
    merged_df = pd.concat([orig_df, label_df], axis=1)
    merged_df.to_csv(label_path, index=True)

if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".avi")]
    for video_file in video_files:
        base = video_file.split('_')[0]  # 例: 20250824141226
        video_path = os.path.join(VIDEO_DIR, video_file)
        label_path = os.path.join(LABEL_DIR, base + "_label.csv")
        if os.path.exists(label_path):
            print(f"ラベリング: {video_file} → {label_path}")
            label_video_and_sensor(video_path, label_path)
        else:
            print(f"label.csvが見つかりません: {label_path}")
