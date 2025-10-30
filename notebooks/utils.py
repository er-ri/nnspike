import fnmatch
import glob
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def view_data_distribution(
    df: pd.DataFrame, cols: list, bins_list: list | None = None
) -> None:
    """Plots the distribution for each column provided in 'cols', with adjustable bins for each column.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        cols (list): List of column names to plot.
        bins_list (list, optional): List of bin counts for each column. If None, defaults to 30 for all.
    """
    sns.set_style("darkgrid", {"grid.color": ".5", "grid.linestyle": ":"})

    n_cols = len(cols)
    plt.figure(figsize=(10 * n_cols, 6))

    if bins_list is None:
        bins_list = [30] * n_cols
    elif len(bins_list) != n_cols:
        raise ValueError("Length of bins_list must match length of cols.")

    for idx, (col, bins) in enumerate(zip(cols, bins_list, strict=False), 1):
        plt.subplot(1, n_cols, idx)
        sns.histplot(df[col], bins=bins, edgecolor=None)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of '{col}'")

    plt.tight_layout()
    plt.show()


def extract_video_frames(video_path: str, frame_path: str) -> None:
    """Extracts frames from a video file and saves them as individual image files.

    Args:
        video_path (str): The path to the input video file.
        frame_path (str): The directory where the extracted frames will be saved.
                          This directory must already exist.

    Raises:
        Exception: If the specified frame_path directory does not exist.

    Example:
        extract_video_frames("input_video.mp4", "output_frames/")
        This will save frames from 'input_video.mp4' into the 'output_frames/' directory
        with filenames like 'frame_1.png', 'frame_2.png', etc.
    """
    cap = cv2.VideoCapture(video_path)  # type: ignore[call-arg]

    path = Path(frame_path)
    if path.is_dir() != True:
        raise Exception("Directory not exists.")

    # Check whether the frame was successfully extracted
    success = 1

    while success:
        success, image = cap.read()

        if not success:
            break

        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Saves the frames with frame-count
        cv2.imwrite(f"{frame_path}frame_{frame_count}.png", image)

    print(f"Frames extracted to: {path.absolute()}")


def get_all_avi_files(
    directory_path: str = "../storage/videos/", filter_timestamp: str | None = None
) -> list[tuple[str, str]]:
    """Get all AVI files from the specified directory with their timestamps.

    Args:
        directory_path (str): Path to the directory containing AVI files
        filter_timestamp (str, optional): If set, only return files matching this timestamp pattern (supports wildcards with *)

    Returns:
        list: List of tuples containing (file_path, timestamp)
    """
    # Use glob to find all .avi files in the directory
    avi_files = glob.glob(os.path.join(directory_path, "*.avi"))
    avi_files = [path.replace("\\", "/") for path in avi_files]

    # Sort the files for consistent ordering
    avi_files.sort()

    # Extract timestamps and create tuples
    result = []
    for avi_file in avi_files:
        # Extract filename without extension
        filename = os.path.basename(avi_file)
        filename_no_ext = os.path.splitext(filename)[0]

        # Extract timestamp from filename (assuming format: timestamp_picamera.avi)
        # This will extract the part before '_picamera'
        timestamp_match = re.match(r"^(\d{14})_.*", filename_no_ext)
        timestamp = timestamp_match.group(1) if timestamp_match else filename_no_ext

        # If filter_timestamp is set, only include matching files using pattern matching
        if filter_timestamp is None or fnmatch.fnmatch(timestamp, filter_timestamp):
            result.append((avi_file, timestamp))

    return result


def extract_frames_from_avi_files(
    avi_files_with_timestamps: list[tuple[str, str]],
    base_output_dir: str = "../storage/frames/",
) -> list[tuple[str, str]]:
    """Extract frames from all AVI files and save them to folders named by timestamp.

    Args:
        avi_files_with_timestamps (list): List of tuples containing (file_path, timestamp)
        base_output_dir (str): Base directory where frame folders will be created

    Returns:
        list: List of tuples containing (output_directory, timestamp)
    """
    output_directories_with_timestamps = []

    for avi_file, timestamp in avi_files_with_timestamps:
        # Create output directory path
        output_dir = os.path.join(base_output_dir, timestamp)

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Add trailing slash for extract_video_frames function
        output_dir_with_slash = output_dir + "/"

        filename = os.path.basename(avi_file)
        print(f"Extracting frames from {filename} to {output_dir_with_slash}")

        try:
            # Extract frames using the nnspike utility function
            extract_video_frames(avi_file, output_dir_with_slash)
            print(f"✓ Successfully extracted frames to {timestamp}/")
            # Add successful output directory and timestamp to the list
            output_directories_with_timestamps.append((output_dir, timestamp))
        except Exception as e:
            print(f"✗ Error extracting frames from {filename}: {str(e)}")

    return output_directories_with_timestamps
