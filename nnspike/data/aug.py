import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
from glob import glob


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
    transform = A.ShiftScaleRotate(
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0,
    )

    # Apply the transformation and get the parameters
    transformed = transform(image=image)
    params = transform.get_params()

    # Extract the transformed image
    transformed_image = transformed["image"]

    return transformed_image, params


def augment_dataset(df, intervals, line_types, p, export_path):
    """Augment images in a dataset based on specified conditions and save them to a new location.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and metadata.
        intervals (list): List of intervals to filter the DataFrame.
        line_types (list): List of line types to filter the DataFrame.
        p (float): Probability threshold for applying augmentation to each row.
        export_path (str): Directory path where augmented images will be saved.

    Returns:
        pd.DataFrame: DataFrame containing metadata of augmented images.

    Example:
        >>> df = pd.read_csv('dataset.csv')
        >>> intervals = [1.0, 1.2]
        >>> line_types = [0,1,2,3]
        >>> p = 0.5
        >>> export_path = 'augmented_images'
        >>> aug_df = augment_dataset(df, intervals, line_types, p, export_path)
    """
    results = list()

    # Filter the DataFrame based on the conditions
    filtered_df = df[
        (df["interval"].isin(intervals))
        & (df["line_type"].isin(line_types))
        & (df["use"] == True)
    ]
    # Apply the random condition
    filtered_df = filtered_df[
        filtered_df.apply(lambda row: random.random() < p, axis=1)
    ]

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        aug_image, _ = random_shift_scale_rotate(
            image, shift_limit=0.05, scale_limit=0.05, rotate_limit=5
        )

        aug_image_path = _convert_aug_image_path(image_path, export_path)

        cv2.imwrite(aug_image_path, aug_image)

        # Append the augmented image path and line type to the results list
        results.append(
            {
                "image_path": aug_image_path,
                "mx": None,
                "predicted_x": None,
                "adjusted_x": None,
                "line_type": row["line_type"],
                "interval": row["interval"],
                "use": False,
                "worker": "aug",
            }
        )

    # Convert the results list to a new DataFrame
    aug_df = pd.DataFrame(results)

    return aug_df


def _convert_aug_image_path(image_path, export_path):
    """Convert the original image path to a new path for the augmented image.

    Args:
        image_path (str): Original image path.
        export_path (str): Directory path where augmented images will be saved.

    Returns:
        str: New path for the augmented image.

    Example:
        >>> image_path = 'images/frame_001.png'
        >>> export_path = 'augmented_images'
        >>> new_path = _convert_aug_image_path(image_path, export_path)
    """
    # Extract the directory and filename from the frame path
    dir_path, filename = image_path.rsplit("\\", 1)

    # Extract the timestamp from the directory path
    _, timestamp = dir_path.rsplit("/", 1)

    # How many times has been augmented
    aug_num = len(glob(f"{timestamp}_{filename.split('.')[0]}_*.png"))

    # Construct the new path
    aug_filename = f"{timestamp}_{filename.split('.')[0]}_{aug_num}.png"

    aug_image_path = f"{export_path}/{aug_filename}"

    return aug_image_path
