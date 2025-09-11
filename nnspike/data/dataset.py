"""
This module defines custom PyTorch Datasets for driving records, including image preprocessing and augmentation.

This module provides dataset classes for different machine learning tasks including regression,
classification, and multi-task learning. All datasets support data augmentation through
brightness/contrast adjustments, RGB shifts, and horizontal flipping based on course direction.

Modules:
    - cv2: OpenCV library for image processing.
    - torch: PyTorch library for tensor operations and neural networks.
    - albumentations as A: Albumentations library for image augmentations.
    - torchvision.transforms as transforms: PyTorch's torchvision library for common image transformations.
    - numpy as np: NumPy library for numerical operations.
    - torch.utils.data.Dataset: Base class for all datasets in PyTorch.
    - nnspike.utils.normalize_image: Custom function for image normalization.
    - nnspike.constants.Mode: Enumeration for different operation modes.

Constants:
    - transform_bright_shift (albumentations.ReplayCompose): Augmentation pipeline with random brightness/contrast adjustments and RGB shifts.
    - transform_flip (albumentations.Compose): Augmentation pipeline for horizontal flipping of images.

Functions:
    - _rand_relative_position(relative_position, position_variation): Adds random variation to relative position for data augmentation.

Classes:
    - RegressionDataset(Dataset): Custom dataset class for regression tasks, predicting continuous values like steering angles.
    - ClassificationDataset(Dataset): Custom dataset class for classification tasks, predicting discrete modes of operation.
    - MultiTaskDataset(Dataset): Custom dataset class for multi-task learning, combining both regression and classification.

RegressionDataset Class:
    Designed for predicting continuous values from images and relative position data.

    Methods:
        - __init__(self, inputs, outputs, roi, train_course, position_variation=0.00625):
            Initializes the dataset with input data, target values, region of interest, training course, and position variation.

        - __len__(self):
            Returns the number of samples in the dataset.

        - __getitem__(self, idx):
            Retrieves and processes the sample at the given index. Returns tuple of (roi_tensor, relative_position) and target_x.

ClassificationDataset Class:
    Designed for predicting discrete modes/classes from images and relative position data.

    Methods:
        - __init__(self, inputs, outputs, roi, train_course, position_variation=0.00625, transform=None):
            Initializes the dataset with input data, class labels, region of interest, training course, position variation, and optional transforms.

        - __len__(self):
            Returns the number of samples in the dataset.

        - __getitem__(self, idx):
            Retrieves and processes the sample at the given index. Returns tuple of (roi_tensor, relative_position) and mode.

MultiTaskDataset Class:
    Designed for simultaneous regression and classification tasks from the same input data.

    Methods:
        - __init__(self, inputs, outputs, roi, train_course, position_variation=0.00625):
            Initializes the dataset with input data, combined outputs (mode, target), region of interest, training course, and position variation.

        - __len__(self):
            Returns the number of samples in the dataset.

        - __getitem__(self, idx):
            Retrieves and processes the sample at the given index. Returns tuple of (roi_tensor, relative_position) and (mode, target_x).

Usage Examples:
    # Regression dataset for predicting steering angles
    regression_dataset = RegressionDataset(
        inputs=[('path/to/image.png', 0.5, 'course1')],
        outputs=[100.0],
        roi=np.array([0, 0, 200, 200]),
        train_course='course1'
    )

    # Classification dataset for predicting operation modes
    classification_dataset = ClassificationDataset(
        inputs=[('path/to/image.png', 0.5, 'course1')],
        outputs=[1],
        roi=np.array([0, 0, 200, 200]),
        train_course='course1'
    )

    # Multi-task dataset for both regression and classification
    multitask_dataset = MultiTaskDataset(
        inputs=[('path/to/image.png', 0.5, 'course1')],
        outputs=[(1, 100.0)],
        roi=np.array([0, 0, 200, 200]),
        train_course='course1'
    )

    # Using with DataLoader
    dataloader = torch.utils.data.DataLoader(regression_dataset, batch_size=4, shuffle=True)
    for (roi_area, relative_position), target in dataloader:
        # Training loop here
        pass

Notes:
    - All datasets apply data augmentation including brightness/contrast adjustments and RGB shifts.
    - Images are horizontally flipped when the course differs from the training course.
    - Relative positions are augmented with random variation for improved generalization.
    - Target values are normalized to [0, 1] range for regression tasks.
    - The `normalize_image` function is used for image preprocessing.
    - ROI (Region of Interest) defines the area of the image to extract for processing.
"""

from __future__ import annotations

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from nnspike.constants import Mode
from nnspike.utils import normalize_image

transform_bright_shift = A.ReplayCompose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    ]
)

transform_flip = A.Compose(
    [A.HorizontalFlip(p=1)],
)


def _rand_relative_position(
    relative_position: float, position_variation: float
) -> torch.Tensor:
    """Add random variation to relative position for data augmentation.

    Args:
        relative_position (float): The original relative position value.
        position_variation (float): The maximum amount of random variation to add/subtract.

    Returns:
        float: The adjusted relative position, clamped between 0 and 1.
    """
    min_val = -position_variation
    max_val = position_variation
    range_val = max_val - min_val

    random_floats = torch.rand(size=[1], dtype=torch.float32)
    random_floats = random_floats * range_val + min_val
    tensor_relative_position = torch.tensor(relative_position, dtype=torch.float32)
    tensor_relative_position = tensor_relative_position + random_floats
    tensor_relative_position = torch.clamp(tensor_relative_position, min=0, max=1)

    return tensor_relative_position


class RegressionDataset(Dataset):
    """Dataset for regression tasks, particularly for predicting continuous values like steering angles.

    This dataset handles image loading, preprocessing, and augmentation for regression tasks.
    It supports data augmentation through random brightness/contrast adjustments and RGB shifts.

    Attributes:
        preprocess (transforms.ToTensor): Transform to convert numpy arrays to PyTorch tensors.
    """

    preprocess = transforms.ToTensor()

    def __init__(
        self,
        inputs: list[tuple[str, float, str]],
        outputs: list[float],
        roi: np.ndarray,
        train_course: str,
        position_variation: float = 0.00625,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.roi = roi
        self.train_course = train_course
        self.position_variation = position_variation

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        image_path = self.inputs[idx][0]
        image = cv2.imread(image_path)
        roi_area = image[self.roi[1] : self.roi[3], self.roi[0] : self.roi[2]]
        relative_position = self.inputs[idx][1]
        course = self.inputs[idx][2]

        target_x = self.outputs[idx] - self.roi[0]

        # Apply brightness and RGB shift
        roi_area = transform_bright_shift(image=roi_area)["image"]

        # Horizontal flip the image if the course is not equal to the training course
        if course != self.train_course:
            roi_area = transform_flip(image=roi_area)["image"]
            target_x = (self.roi[2] - self.roi[0]) - target_x

        roi_area = normalize_image(image=roi_area)
        tensor_roi_area = self.preprocess(roi_area)  # Convert to pytorch tensor
        tensor_roi_area = tensor_roi_area.to(
            torch.float32
        )  # `Conv2d` supports up to `float32`

        # Add the random floats to the original relative position
        relative_position = _rand_relative_position(
            relative_position, self.position_variation
        )

        target_x = (target_x) / (self.roi[2] - self.roi[0])
        target_x = torch.tensor(target_x, dtype=torch.float32).unsqueeze(-1)

        return (tensor_roi_area, relative_position), target_x


class ClassificationDataset(Dataset):
    """Dataset for classification tasks, particularly for predicting discrete modes of operation.

    This dataset handles image loading, preprocessing, and augmentation for classification tasks.
    It supports data augmentation through random brightness/contrast adjustments, RGB shifts,
    and optional custom transforms.

    Attributes:
        preprocess (transforms.ToTensor): Transform to convert numpy arrays to PyTorch tensors.
    """

    preprocess = transforms.ToTensor()

    def __init__(
        self,
        inputs: list[tuple[str, float, str]],
        outputs: list[int],
        roi: np.ndarray,
        train_course: str,
        position_variation: float = 0.00625,
        transform: A.Transform | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.roi = roi
        self.train_course = train_course
        self.position_variation = position_variation
        self.transform = transform

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        image_path = self.inputs[idx][0]
        image = cv2.imread(image_path)
        roi_area = image[self.roi[1] : self.roi[3], self.roi[0] : self.roi[2]]
        relative_position = self.inputs[idx][1]
        course = self.inputs[idx][2]

        # Apply brightness and RGB shift
        roi_area = transform_bright_shift(image=roi_area)["image"]

        # Horizontal flip the image if the course is not equal to the training course
        if course != self.train_course:
            roi_area = transform_flip(image=roi_area)["image"]

        # Perform additional transform if provided
        if self.transform:
            roi_area = self.transform(image=roi_area)["image"]

        roi_area = normalize_image(image=roi_area)
        tensor_roi_area = self.preprocess(roi_area)  # Convert to pytorch tensor
        tensor_roi_area = tensor_roi_area.to(
            torch.float32
        )  # `Conv2d` supports up to `float32`

        # Add the random floats to the original relative position
        relative_position = _rand_relative_position(
            relative_position, self.position_variation
        )

        # Convert the mode to a tensor
        mode = torch.tensor(self.outputs[idx], dtype=torch.long)

        return (tensor_roi_area, relative_position), mode


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning, combining both regression and classification tasks.

    This dataset handles image loading, preprocessing, and augmentation for both regression
    and classification tasks simultaneously. It supports data augmentation through random
    brightness/contrast adjustments and RGB shifts.

    Attributes:
        preprocess (transforms.ToTensor): Transform to convert numpy arrays to PyTorch tensors.
    """

    preprocess = transforms.ToTensor()

    def __init__(
        self,
        inputs: list[tuple[str, float, str]],
        outputs: list[tuple[int, float]],
        roi: np.ndarray,
        train_course: str,
        position_variation: float = 0.00625,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.roi = roi
        self.train_course = train_course
        self.position_variation = position_variation

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, torch.Tensor]]:
        image_path = self.inputs[idx][0]
        image = cv2.imread(image_path)
        roi_area = image[self.roi[1] : self.roi[3], self.roi[0] : self.roi[2]]
        relative_position = self.inputs[idx][1]
        course = self.inputs[idx][2]

        target_x = self.outputs[idx][1] - self.roi[0]
        mode = self.outputs[idx][0]

        # Apply brightness and RGB shift
        roi_area = transform_bright_shift(image=roi_area)["image"]

        # Horizontal flip the image if the course is not equal to the training course
        if course != self.train_course:
            roi_area = transform_flip(image=roi_area)["image"]
            target_x = (self.roi[2] - self.roi[0]) - target_x
            mode = (
                Mode.FOLLOW_LEFT_EDGE.value
                if mode == Mode.FOLLOW_RIGHT_EDGE.value
                else Mode.FOLLOW_RIGHT_EDGE.value
            )

        roi_area = normalize_image(image=roi_area)
        tensor_roi_area = self.preprocess(roi_area)  # Convert to pytorch tensor
        tensor_roi_area = tensor_roi_area.to(
            torch.float32
        )  # `Conv2d` supports up to `float32`

        # Add the random floats to the original relative position
        relative_position = _rand_relative_position(
            relative_position, self.position_variation
        )

        target_x = (target_x) / (self.roi[2] - self.roi[0])
        target_x = torch.tensor(target_x, dtype=torch.float32).unsqueeze(-1)

        return (tensor_roi_area, relative_position), (mode, target_x)
