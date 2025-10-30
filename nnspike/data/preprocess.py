"""This module provides functions for preprocessing and preparing datasets of image frames with sensor data.

The module contains utilities for creating labeled datasets from image files, balancing data distributions,
sorting by frame numbers, and merging sensor status data with image metadata. It's designed to work with
robot control data that includes motor positions, sensor readings, and image frames.

Functions:
    balance_dataset(df: pd.DataFrame, col_name: str, max_samples: int, num_bins: int) -> pd.DataFrame:
        Balances the dataset by limiting the number of samples in each bin of a specified column.
        Uses histogram binning to ensure uniform distribution across value ranges.

    sort_by_frames_number(df: pd.DataFrame) -> pd.DataFrame:
        Sorts a DataFrame by the frame number extracted from the 'image_path' column and adds
        the frame_number column as the second column in the DataFrame.

    create_label_dataframe(path_pattern: str, course: str) -> pd.DataFrame:
        Creates a comprehensive DataFrame with image paths and associated metadata columns including
        motor speeds, positions, sensor readings, and labeling fields. Initializes all sensor
        columns with default values for subsequent population.

    set_spike_status(label_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
        Merges comprehensive sensor and motor data from status_df into label_df based on matching
        frame numbers. Updates multiple columns including motor speeds, positions, and various
        sensor readings (distance, color sensors).
"""

import random
from glob import glob

import numpy as np
import pandas as pd

from nnspike.constants import OFFSET_Y


def balance_dataset(
    df: pd.DataFrame, col_name: str, max_samples: int, num_bins: int
) -> pd.DataFrame:
    """Balances the dataset by limiting the number of samples in each bin of a specified column.

    This function creates a histogram of the specified column and ensures that no bin has more than
    `max_samples` samples. If a bin exceeds this limit, excess samples are randomly removed to balance
    the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be balanced.
        col_name (str): The name of the column to be used for creating bins.
        max_samples (int): The maximum number of samples allowed per bin.
        num_bins (int): The number of bins to divide the column into.

    Returns:
        pd.DataFrame: A DataFrame with the dataset balanced according to the specified column and bin limits.

    Note:
        Make sure the column does not have
            1. None/Nan
            2. empty string
        Otherwise, `ValueError: autodetected range of [nan, nan] is not finite` may raise
    """
    # Reset index to ensure clean 0,1,2,3... sequence
    df = df.reset_index(drop=True)

    # Create bins using pd.cut
    bins_series = pd.cut(df[col_name], bins=num_bins, include_lowest=True)

    # Initialize an empty list to store indices to remove
    remove_list = []

    for bin_label in bins_series.cat.categories:
        bin_indices = df[bins_series == bin_label].index.tolist()

        if len(bin_indices) > max_samples:
            random.shuffle(bin_indices)
            remove_num = len(bin_indices) - max_samples
            remove_list.extend(bin_indices[:remove_num])

    df_filtered = df.drop(remove_list)
    return df_filtered


def sort_by_frames_number(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts a DataFrame by the frame number extracted from the 'image_path' column.

    This function extracts the frame number from the 'image_path' column of the
    DataFrame, sorts the DataFrame based on these frame numbers, and keeps
    the 'frame_number' column as the 2nd column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing an 'image_path' column
            with file paths that include frame numbers in the format 'frame_<number>'.

    Returns:
        pd.DataFrame: The sorted DataFrame with rows ordered by the extracted
        frame numbers and the frame_number column as the 2nd column.
    """
    df["frame_number"] = (
        df["image_path"].str.extract(r"frame_(\d+)", expand=False).astype(int)
    )

    # Sort the DataFrame by the extracted frame number
    df_sorted = df.sort_values(by="frame_number")

    # Reorder columns to put frame_number as the 2nd column
    columns = df_sorted.columns.tolist()
    if "frame_number" in columns:
        columns.remove("frame_number")
        columns.insert(1, "frame_number")
        df_sorted = df_sorted[columns]

    df_sorted = df_sorted.reset_index(drop=True)

    return df_sorted


def create_label_dataframe(path_pattern: str, course: str) -> pd.DataFrame:
    """Creates a comprehensive DataFrame with image paths and associated metadata for labeling tasks.

    This function searches for image files matching the given path pattern and constructs
    a DataFrame containing the paths to these images along with multiple columns for sensor
    data and robot control information. The DataFrame includes columns for:
    - Image metadata: image_path, course, data_type, use flag
    - Target and mode information: mode, target_x
    - Motor data: motor_a_speed, motor_b_speed, motor_a_relative_position, motor_b_relative_position
    - Sensor readings: distance_sensor, color_reflected, color_ambient, color_value

    All sensor and motor columns are initialized with default values (0 for numeric fields,
    NaN for mode and target_x, True for use flag) to be populated later by other functions.

    Args:
        path_pattern (str): A glob pattern to match image file paths.
        course (str): The name of the course associated with the images.

    Returns:
        pd.DataFrame: A DataFrame containing the image paths and comprehensive metadata columns
        with default values for subsequent data population.
    """
    image_paths = glob(path_pattern)
    # Convert backslashes to forward slashes for consistent path formatting
    image_paths = [path.replace("\\", "/") for path in image_paths]

    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "mode": [np.nan] * len(image_paths),
            "target_x": [np.nan] * len(image_paths),
            "motor_a_speed": [0] * len(image_paths),
            "motor_b_speed": [0] * len(image_paths),
            "motor_a_relative_position": [0] * len(image_paths),
            "motor_b_relative_position": [0] * len(image_paths),
            "distance_sensor": [0] * len(image_paths),
            "color_reflected": [0] * len(image_paths),
            "color_ambient": [0] * len(image_paths),
            "color_value": [0] * len(image_paths),
            "data_type": [0] * len(image_paths),
            "course": [course] * len(image_paths),
            "use": [True] * len(image_paths),
        }
    )

    return df


def set_spike_status(label_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
    """Merges comprehensive sensor and motor data from status_df into label_df based on matching frame numbers.

    This function performs a comprehensive merge of robot sensor and control data from status_df
    into label_df by matching frame_number values. It updates multiple columns including:
    - Motor control: motor_a_speed, motor_b_speed, motor_a_relative_position, motor_b_relative_position
    - Sensor readings: distance_sensor, color_reflected, color_ambient, color_value
    - Robot mode: mode (preserving manual labels from label_df when available)

    The merge is performed as a left join, preserving all rows in label_df and only updating
    sensor/motor data where matching frame numbers exist in status_df. If label_df doesn't
    have a frame_number column, it will be automatically added by calling sort_by_frames_number().

    Args:
        label_df (pd.DataFrame): The label DataFrame containing image paths and metadata.
            Must have or be able to generate a frame_number column.
        status_df (pd.DataFrame): The status DataFrame containing comprehensive sensor and motor
            data with frame_number column for matching.

    Returns:
        pd.DataFrame: The updated label DataFrame with sensor and motor data merged from status_df.
        All original columns are preserved, and sensor data is filled where frame matches exist.
    """
    # Ensure frame_number column exists in label_df
    if "frame_number" not in label_df.columns:
        label_df = sort_by_frames_number(label_df)

    # Select relevant columns from status_df for merging
    spike_columns = [
        "frame_number",
        "mode",
        "motor_a_speed",
        "motor_b_speed",
        "motor_a_relative_position",
        "motor_b_relative_position",
        "distance_sensor",
        "color_reflected",
        "color_ambient",
        "color_value",
    ]
    spike_subset = status_df[spike_columns].copy()

    # Merge the DataFrames on frame_number, updating motor position columns
    # Use left join to preserve all rows in label_df
    merged_df = label_df.merge(
        spike_subset, on="frame_number", how="left", suffixes=("", "_temp")
    )

    # Update the motor position columns where spike data is available
    merged_df["motor_a_relative_position"] = merged_df[
        "motor_a_relative_position_temp"
    ].fillna(merged_df["motor_a_relative_position"])
    merged_df["motor_b_relative_position"] = merged_df[
        "motor_b_relative_position_temp"
    ].fillna(merged_df["motor_b_relative_position"])
    merged_df["mode"] = label_df["mode"].fillna(
        merged_df["mode_temp"]
    )  # Use label_df's mode which was set by label_dataset_by_opencv
    merged_df["motor_a_speed"] = merged_df["motor_a_speed_temp"].fillna(
        merged_df["motor_a_speed"]
    )
    merged_df["motor_b_speed"] = merged_df["motor_b_speed_temp"].fillna(
        merged_df["motor_b_speed"]
    )
    merged_df["distance_sensor"] = merged_df["distance_sensor_temp"].fillna(
        merged_df["distance_sensor"]
    )
    merged_df["color_reflected"] = merged_df["color_reflected_temp"].fillna(
        merged_df["color_reflected"]
    )
    merged_df["color_ambient"] = merged_df["color_ambient_temp"].fillna(
        merged_df["color_ambient"]
    )
    merged_df["color_value"] = merged_df["color_value_temp"].fillna(
        merged_df["color_value"]
    )

    # Drop the temporary spike columns
    columns_to_drop = [col for col in merged_df.columns if col.endswith("_temp")]
    merged_df = merged_df.drop(columns=columns_to_drop)

    return merged_df
