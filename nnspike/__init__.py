"""nnspike: A LEGO SPIKE Robot using Neural Network for line following.

This package provides tools and utilities for controlling a LEGO SPIKE robot
using neural network-based computer vision for line following tasks.

The package includes:
    - Unit control classes for robot communication and management
    - Neural network models for vision-based navigation
    - Data processing and augmentation utilities
    - Image processing and control algorithms
"""

from . import constants, data, models, unit, utils

__version__ = "2025.09"
__author__ = "er-ri"

__all__ = ["constants", "data", "models", "unit", "utils"]
