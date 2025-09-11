"""Data processing and dataset management for nnspike.

This module provides comprehensive data handling capabilities for neural network
training with the LEGO SPIKE robot, including dataset classes, data augmentation,
and preprocessing utilities.

Modules:
    aug: Data augmentation functions for improving model robustness
    dataset: PyTorch dataset classes for different learning tasks
    preprocess: Data preprocessing and label management utilities

Example:
    Basic usage for creating and augmenting a dataset:

        from nnspike.data import RegressionDataset, augment_dataset

        # Create a dataset
        dataset = RegressionDataset(data_dir="path/to/data")

        # Apply augmentation
        augmented_data = augment_dataset(dataset, factor=2)
"""

from .aug import augment_dataset, random_shift_scale_rotate
from .dataset import ClassificationDataset, MultiTaskDataset, RegressionDataset
from .preprocess import (
    balance_dataset,
    create_label_dataframe,
    set_spike_status,
    sort_by_frames_number,
)

__all__ = [
    # Augmentation
    "random_shift_scale_rotate",
    "augment_dataset",
    # Dataset
    "RegressionDataset",
    "ClassificationDataset",
    "MultiTaskDataset",
    # Preprocessing
    "balance_dataset",
    "sort_by_frames_number",
    "create_label_dataframe",
    "set_spike_status",
]
