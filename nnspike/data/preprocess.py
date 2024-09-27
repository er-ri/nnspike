"""
This module provides functions for labeling and balancing datasets of image frames.

Functions:
    label_dataset_by_opencv(df: pd.DataFrame, interval: int, line_types: list[int], use: bool, worker: str, roi: tuple[int, int, int, int]) -> pd.DataFrame:
        Labels all the frame files in the provided DataFrame using OpenCV and updates the 'mx' and 'my' columns with the calculated values.
        
    label_dataset_by_model(df: pd.DataFrame, model: torch.nn.Module, intervals: list[int], line_types: list[int], use: bool, worker: str, roi: tuple[int, int, int, int]) -> pd.DataFrame:
        Deprecated. Labels all the frame files in the provided DataFrame using a deep learning model and updates the 'predicted_x' column with the calculated values.
        
    balance_dataset(df: pd.DataFrame, col_name: str, max_samples: int, num_bins: int) -> pd.DataFrame:
        Balances the dataset by limiting the number of samples in each bin of a specified column.
        
    sort_by_frames_number(df: pd.DataFrame) -> pd.DataFrame:
        Sorts a DataFrame by the frame number extracted from the 'file_path' column.
        
    create_label_csv(path_pattern: str, course: str, worker: str) -> pd.DataFrame:
        Creates a DataFrame with initial labels for images matching the given path pattern.
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
    df, interval, line_types, use, worker, roi: tuple[int, int, int, int]
) -> pd.DataFrame:
    """
    Labels all the frame files in the provided DataFrame using OpenCV and updates the 'mx' and 'my' columns with the calculated values.

    Args:
        df (pd.DataFrame): The input DataFrame containing image paths and metadata.
        interval (int): The interval to filter the DataFrame.
        line_types (list[int]): The list of line types to filter the DataFrame.
        use (bool): The usage flag to filter the DataFrame.
        worker (str): The worker identifier to filter the DataFrame.
        roi (tuple[int, int, int, int]): The region of interest (ROI) in the format (x1, y1, x2, y2).

    Returns:
        pd.DataFrame: The updated DataFrame with 'mx' columns labeled.
    """
    x1, y1, x2, y2 = roi

    filtered_df = df[
        (df["interval"] == interval)
        & (df["line_type"].isin(line_types))
        & (df["use"] == use)
        & (df["worker"] == worker)
    ]

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


def label_dataset_by_model(df, model, intervals, line_types, use, worker, roi):
    """
    Deprecated.
    Labels all the frame files in the provided DataFrame using a deep learning model and
    updates the 'predicted_x' column with the calculated values.

    Args:
        df (pd.DataFrame): The input DataFrame containing image paths and metadata.
        model (torch.nn.Module): The deep learning model used for labeling.
        intervals (list[int]): The list of intervals to filter the DataFrame.
        line_types (list[int]): The list of line types to filter the DataFrame.
        use (bool): The usage flag to filter the DataFrame.
        worker (str): The worker identifier to filter the DataFrame.
        roi (tuple[int, int, int, int]): The region of interest (ROI) in the format (x1, y1, x2, y2).

    Returns:
        pd.DataFrame: The updated DataFrame with 'predicted_x' column labeled.
    """
    x1, y1, x2, y2 = roi
    transform = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Filter the DataFrame based on the conditions before iterating
    filtered_df = df[
        (df["interval"].isin(intervals))
        & (df["line_type"].isin(line_types))
        & (df["use"] == use)
        & (df["worker"] == worker)
    ]

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

    hist, bins = np.histogram(df[col_name], num_bins)

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

    This function extracts the frame number from the 'file_path' column of the
    DataFrame, sorts the DataFrame based on these frame numbers, and then drops
    the temporary 'frame_number' column before returning the sorted DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'file_path' column
            with file paths that include frame numbers in the format 'frame_<number>'.

    Returns:
        pd.DataFrame: The sorted DataFrame with rows ordered by the extracted
        frame numbers.
    """
    df["frame_number"] = (
        df["image_path"].str.extract(r"frame_(\d+)", expand=False).astype(int)
    )

    # Sort the DataFrame by the extracted frame number
    df_sorted = df.sort_values(by="frame_number")

    df_sorted = df_sorted.drop(columns=["frame_number"])
    df_sorted = df_sorted.reset_index(drop=True)

    return df_sorted


def create_label_df(path_pattern: str, course: str, worker: str) -> pd.DataFrame:
    """
    Creates a DataFrame with image paths and associated metadata for labeling tasks.

    This function searches for image files matching the given path pattern and constructs
    a DataFrame containing the paths to these images along with several columns initialized
    with default values. The DataFrame includes columns for manual x-coordinates (`mx`),
    predicted x-coordinates (`predicted_x`), adjusted x-coordinates (`adjusted_x`), line type,
    course name, interval, usage flag, and worker identifier.

    Args:
        path_pattern (str): A glob pattern to match image file paths.
        course (str): The name of the course associated with the images.
        worker (str): The identifier of the worker who will label the images.

    Returns:
        pd.DataFrame: A DataFrame containing the image paths and associated metadata.
    """
    image_paths = glob(path_pattern)

    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "mx": [np.nan] * len(image_paths),
            "predicted_x": [np.nan] * len(image_paths),
            "adjusted_x": [np.nan] * len(image_paths),
            "line_type": [0] * len(image_paths),
            "course": [course] * len(image_paths),
            "interval": [1] * len(image_paths),
            "use": [True] * len(image_paths),
            "worker": [worker] * len(image_paths),
        }
    )

    return df
