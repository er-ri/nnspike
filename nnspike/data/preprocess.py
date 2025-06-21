"""
This module provides functions for labeling and balancing datasets of image frames.

Functions:
    label_dataset_by_opencv(df: pd.DataFrame, data_types: list, roi: tuple[int, int, int, int]) -> pd.DataFrame:
        Labels all the frame files in the provided DataFrame using OpenCV and updates the 'mx' column with the calculated values.

    label_dataset_by_model(df: pd.DataFrame, model: torch.nn.Module, data_types: list, roi: tuple[int, int, int, int]) -> pd.DataFrame:
        Deprecated. Labels all the frame files in the provided DataFrame using a deep learning model and updates the 'predicted_x' column with the calculated values.

    balance_dataset(df: pd.DataFrame, col_name: str, max_samples: int, num_bins: int) -> pd.DataFrame:
        Balances the dataset by limiting the number of samples in each bin of a specified column.

    sort_by_frames_number(df: pd.DataFrame) -> pd.DataFrame:
        Sorts a DataFrame by the frame number extracted from the 'image_path' column.

    create_label_df(path_pattern: str, course: str) -> pd.DataFrame:
        Creates a DataFrame with image paths and associated metadata for labeling tasks.

    set_spike_status(label_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
        Sets the motor position values in label_df from status_df based on matching frame numbers.
"""

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
from glob import glob
from sklearn.utils import shuffle
from nnspike.utils import steer_by_camera, normalize_image


def label_dataset_by_opencv(
    df, data_types, roi: tuple[int, int, int, int]
) -> pd.DataFrame:
    """
    Labels all the frame files in the provided DataFrame using OpenCV and updates the 'mx' column with the calculated values.

    Args:
        df (pd.DataFrame): The input DataFrame containing image paths and metadata.
        data_types (list): The list of data types to filter the DataFrame.
        roi (tuple[int, int, int, int]): The region of interest (ROI) in the format (x1, y1, x2, y2).

    Returns:
        pd.DataFrame: The updated DataFrame with 'mx' column labeled.
    """
    x1, y1, x2, y2 = roi

    filtered_df = df[(df["data_type"].isin(data_types)) & (df["use"] == True)]

    for index, row in tqdm(
        filtered_df.iterrows(), total=len(filtered_df), desc="Processing"
    ):
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        roi_area = gray[y1:y2, x1:x2]
        mx, _, _ = steer_by_camera(image=roi_area)

        # Add `trace_x` to column 'predicted_x'
        df.at[index, "mx"] = x1 + mx

    return df


def label_dataset_by_model(df, model, data_types, roi):
    """
    Deprecated.
    Labels all the frame files in the provided DataFrame using a deep learning model and
    updates the 'predicted_x' column with the calculated values.

    Args:
        df (pd.DataFrame): The input DataFrame containing image paths and metadata.
        model (torch.nn.Module): The deep learning model used for labeling.
        data_types (list): The list of data types to filter the DataFrame.
        roi (tuple[int, int, int, int]): The region of interest (ROI) in the format (x1, y1, x2, y2).

    Returns:
        pd.DataFrame: The updated DataFrame with 'predicted_x' column labeled.
    """
    x1, y1, x2, y2 = roi
    transform = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Filter the DataFrame based on the conditions before iterating
    filtered_df = df[(df["data_type"].isin(data_types)) & (df["use"] == True)]

    for index, row in tqdm(
        filtered_df.iterrows(), total=len(filtered_df), desc="Processing"
    ):
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        roi_area = image[y1:y2, x1:x2]
        roi_area = normalize_image(image=roi_area)
        roi_area = transform(roi_area)
        roi_area = roi_area.to(torch.float32)
        roi_area = roi_area.unsqueeze(0)
        roi_area = roi_area.to(device)

        with torch.no_grad():
            output = model(roi_area)

        trace_x = (output * (x2 - x1))[0][0].detach().item()
        # Add `trace_x` to column 'predicted_x'
        df.at[index, "predicted_x"] = trace_x

    return df


def balance_dataset(
    df: pd.DataFrame, col_name: str, max_samples: int, num_bins: int
) -> pd.DataFrame:
    """
    Balances the dataset by limiting the number of samples in each bin of a specified column.

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

    _, bins = np.histogram(df[col_name], num_bins)

    # Initialize an empty list to store indices to remove
    remove_list = list()

    # Iterate over each bin
    for i in range(num_bins):
        # Get the indices of the samples in the current bin
        bin_indices = df[
            (df[col_name] >= bins[i]) & (df[col_name] <= bins[i + 1])
        ].index.tolist()

        # Shuffle the indices
        bin_indices = shuffle(bin_indices)

        # If the number of samples in the bin exceeds the limit, add the excess to the remove list
        if len(bin_indices) > max_samples:
            remove_list.extend(bin_indices[max_samples:])

    # Drop the rows from the DataFrame
    df = df.drop(remove_list)

    return df


def sort_by_frames_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts a DataFrame by the frame number extracted from the 'image_path' column.

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


def create_label_df(path_pattern: str, course: str) -> pd.DataFrame:
    """
    Creates a DataFrame with image paths and associated metadata for labeling tasks.

    This function searches for image files matching the given path pattern and constructs
    a DataFrame containing the paths to these images along with several columns initialized
    with default values. The DataFrame includes columns for manual x-coordinates (`mx`),
    predicted x-coordinates (`predicted_x`), adjusted x-coordinates (`adjusted_x`),
    course name, relative position, data type, and usage flag.

    Args:
        path_pattern (str): A glob pattern to match image file paths.
        course (str): The name of the course associated with the images.

    Returns:
        pd.DataFrame: A DataFrame containing the image paths and associated metadata.
    """
    image_paths = glob(path_pattern)
    # Convert backslashes to forward slashes for consistent path formatting
    image_paths = [path.replace("\\", "/") for path in image_paths]

    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "mx": [np.nan] * len(image_paths),
            "predicted_x": [np.nan] * len(image_paths),
            "adjusted_x": [np.nan] * len(image_paths),
            "course": [course] * len(image_paths),
            "motor_a_relative_position": [0] * len(image_paths),
            "motor_b_relative_position": [0] * len(image_paths),
            "data_type": [0] * len(image_paths),
            "use": [True] * len(image_paths),
        }
    )

    return df


def set_spike_status(label_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the motor position values in label_df from status_df based on matching frame numbers.

    This function merges motor position data (motor_a_relative_position and motor_b_relative_position)
    from status_df into label_df by matching frame_number values. The merge is performed as a left join,
    preserving all rows in label_df and only updating motor positions where matching frame numbers exist.

    Args:
        label_df (pd.DataFrame): The label DataFrame containing image paths and metadata.
        status_df (pd.DataFrame): The spike DataFrame containing motor position data with frame numbers.

    Returns:
        pd.DataFrame: The updated label DataFrame with motor position values set from status_df.
    """
    # Ensure frame_number column exists in label_df
    if "frame_number" not in label_df.columns:
        label_df = sort_by_frames_number(label_df)

    # Select relevant columns from status_df for merging
    spike_columns = [
        "frame_number",
        "motor_a_relative_position",
        "motor_b_relative_position",
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

    # Drop the temporary spike columns
    columns_to_drop = [col for col in merged_df.columns if col.endswith("_temp")]
    merged_df = merged_df.drop(columns=columns_to_drop)

    return merged_df
