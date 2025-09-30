"""Image augmentation utilities for neural network training data.

This module provides functions for applying various image transformations to augment
training datasets. It includes geometric transformations such as shift, scale, rotation,
and perspective distortion to improve model robustness and generalization.

The module is designed to work with computer vision datasets, particularly for robot
navigation and control tasks. It uses the Albumentations library for efficient and
reproducible image transformations.

Examples:
    Basic usage for single image transformations:

        >>> import numpy as np
        >>> from nnspike.data.aug import (
        ...     random_shift_scale_rotate,
        ...     perspective_transform,
        ... )
        >>>
        >>> # Load an image (example with random data)
        >>> image = np.random.rand(224, 224, 3).astype(np.uint8)
        >>>
        >>> # Apply shift, scale, and rotation
        >>> aug_image, params = random_shift_scale_rotate(image)
        >>>
        >>> # Apply perspective transformation
        >>> aug_image = perspective_transform(aug_image)

    Dataset augmentation workflow:

        >>> import pandas as pd
        >>> from nnspike.data.aug import augment_dataset
        >>>
        >>> # Prepare DataFrame with image metadata
        >>> df = pd.DataFrame(
        ...     {
        ...         "image_path": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
        ...         "use": [True, True],
        ...         "frame_number": [1, 2],
        ...         "mode": ["train", "train"],
        ...         # ... other required columns
        ...     }
        ... )
        >>>
        >>> # Augment 50% of the usable images
        >>> augmented_df = augment_dataset(df, p=0.5, export_path="/path/to/augmented")

Attributes:
    This module contains the following main functions:
    - random_shift_scale_rotate: Applies geometric transformations with controllable parameters
    - perspective_transform: Applies perspective distortion to simulate camera angle changes
    - augment_dataset: Batch processes DataFrame of images with augmentation pipeline

Note:
    All augmentation functions preserve the original image format and dimensions unless
    specifically configured otherwise. The transformations are designed to maintain
    realistic image characteristics suitable for robot vision tasks.

Dependencies:
    - albumentations: For efficient image transformations
    - opencv-python (cv2): For image I/O operations
    - numpy: For array operations
    - pandas: For dataset metadata management
    - tqdm: For progress tracking during batch operations
"""

import os
import random
from typing import cast

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def random_shift_scale_rotate(
    image: np.ndarray,
    shift_limit: float = 0.0625,
    scale_limit: float = 0.1,
    rotate_limit: int = 15,
) -> tuple:
    """Apply a random shift, scale, and rotation transformation to an input image.

    Args:
        image (np.ndarray): The input image to be transformed.
        shift_limit (float, optional): Maximum fraction of total height/width to shift the image. Default is 0.0625.
        scale_limit (float, optional): Maximum scaling factor. Default is 0.1.
        rotate_limit (int, optional): Maximum rotation angle in degrees. Default is 15.

    Returns:
        tuple: A tuple containing:
            - transformed_image (np.ndarray): The transformed image.
            - params (dict): The parameters used for the transformation.

    Example:
        >>> import numpy as np
        >>> image = np.random.rand(100, 100, 3)
        >>> transformed_image, params = random_shift_scale_rotate(image)
    """
    transform = A.Affine(
        translate_percent={
            "x": (-shift_limit, shift_limit),
            "y": (-shift_limit, shift_limit),
        },
        scale=(1 - scale_limit, 1 + scale_limit),
        rotate=(-rotate_limit, rotate_limit),
        border_mode=cv2.BORDER_REFLECT,
        p=1.0,
    )

    # Apply the transformation and get the parameters
    transformed = transform(image=image)
    params = transform.get_params()

    # Extract the transformed image
    transformed_image = transformed["image"]

    return transformed_image, params


def perspective_transform(
    image: np.ndarray,
    scale: tuple = (0.01, 0.05),
    keep_size: bool = True,
) -> np.ndarray:
    """Apply a perspective transformation to an input image.

    Args:
        image (np.ndarray): The input image to be transformed.
        scale (tuple, optional): Range for perspective distortion scale. Default is (0.01, 0.05).
        keep_size (bool, optional): Whether to keep the original image size. Default is True.

    Returns:
        np.ndarray: The transformed image.

    Example:
        >>> import numpy as np
        >>> image = np.random.rand(100, 100, 3)
        >>> transformed_image = perspective_transform(image)
    """
    transform = A.Compose([A.Perspective(scale=scale, keep_size=keep_size, p=1.0)])

    # Apply the transformation and get the parameters
    transformed = transform(image=image)

    # Extract the transformed image
    transformed_image = cast(np.ndarray, transformed["image"])

    return transformed_image


def augment_dataset(df: pd.DataFrame, p: float, export_path: str) -> pd.DataFrame:
    """Augment images in a dataset based on specified conditions and save them to a new location.

    This function filters a DataFrame to include only rows where 'use' is True, then randomly
    selects rows based on probability p. For each selected row, it loads the image, applies
    sequential transformations (shift-scale-rotate followed by perspective transform), and
    saves the augmented image to the export directory. The function returns a new DataFrame
    containing metadata for all augmented images.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and metadata. Must include columns:
            'image_path', 'use', 'frame_number', 'mode', 'course', 'motor_a_relative_position',
            'motor_b_relative_position', 'data_type'.
        p (float): Probability threshold (0.0-1.0) for applying augmentation to each row
            where 'use' is True. Higher values result in more augmented images.
        export_path (str): Directory path where augmented images will be saved. Must not
            already exist - the function will create it.

    Returns:
        pd.DataFrame: DataFrame containing metadata of augmented images with the same
            structure as the input DataFrame. The 'image_path' column will contain paths
            to the newly created augmented images, and 'target_x', 'left_x', 'right_x'
            will be set to None.

    Raises:
        FileExistsError: If the export_path directory already exists.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "image_path": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        ...         "use": [True, True],
        ...         "frame_number": [1, 2],
        ...         "mode": ["A", "B"],
        ...         "course": ["course1", "course1"],
        ...         "motor_a_relative_position": [0, 10],
        ...         "motor_b_relative_position": [0, -5],
        ...         "data_type": ["train", "train"],
        ...     }
        ... )
        >>> augmented_df = augment_dataset(df, p=0.5, export_path="/path/to/augmented")

    Note:
        The function applies two sequential transformations:
        1. Random shift, scale, rotate with limits: shift_limit=0.05, scale_limit=0.05, rotate_limit=5
        2. Perspective transform with scale=(0.01, 0.05)
    """
    # Check if the export directory already exists and throw an error
    if os.path.exists(export_path):
        raise FileExistsError(
            f"Export directory '{export_path}' already exists. Please choose a different path or remove the existing directory."
        )

    # Create the export directory
    os.makedirs(export_path, exist_ok=False)

    results = []  # Filter the DataFrame based on the conditions
    filtered_df = df[(df["use"] == True)]
    # Apply the random condition
    filtered_df = filtered_df[
        filtered_df.apply(lambda row: random.random() < p, axis=1)
    ]

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        image_path = row["image_path"]
        image = cv2.imread(image_path)

        # Apply both transformations sequentially
        # First apply shift, scale, rotate transformation
        aug_image, _ = random_shift_scale_rotate(
            image=image, shift_limit=0.05, scale_limit=0.05, rotate_limit=5
        )

        # Then apply perspective transformation
        aug_image = perspective_transform(
            image=aug_image, scale=(0.01, 0.05), keep_size=True
        )

        _, filename = image_path.rsplit("/", 1)
        aug_image_path = f"{export_path}/{filename}"

        cv2.imwrite(aug_image_path, aug_image)

        # Append the augmented image path and line type to the results list
        results.append(
            {
                "image_path": aug_image_path,
                "frame_number": row["frame_number"],
                "target_x": None,
                "left_x": None,
                "right_x": None,
                "mode": row["mode"],
                "course": row["course"],
                "motor_a_relative_position": row["motor_a_relative_position"],
                "motor_b_relative_position": row["motor_b_relative_position"],
                "data_type": row["data_type"],
                "use": True,
            }
        )

    # Convert the results list to a new DataFrame
    aug_df = pd.DataFrame(results)

    return aug_df
