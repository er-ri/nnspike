"""Neural network models for vision-based robot navigation.

This module contains neural network architectures designed for the LEGO SPIKE
robot's line following and navigation tasks. It includes both custom and adapted
models for different learning approaches.

Modules:
    customized: Custom neural network architectures for specific tasks
    loss: Custom loss functions for multi-task learning
    nvidia: Adapted NVIDIA models for regression and multi-task learning

Example:
    Creating and using a regression model:

        from nnspike.models import NvidiaModelRegression

        # Initialize model
        model = NvidiaModelRegression()

        # Use model for inference
        prediction = model(input_tensor)
"""

from .customized import SimpleNetClassification25
from .loss import MultiTaskLoss
from .nvidia import NvidiaModelMultiTask, NvidiaModelRegression

__all__ = [
    "SimpleNetClassification25",
    "NvidiaModelMultiTask",
    "NvidiaModelRegression",
    "MultiTaskLoss",
]
