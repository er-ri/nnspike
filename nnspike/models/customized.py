import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetClassification25(nn.Module):
    """A simple convolutional neural network for processing image data with additional sensor inputs.

    This network consists of two convolutional layers followed by max pooling, and two fully
    connected layers. It accepts both image data and relative position information as inputs.
    The architecture is designed to handle input images and concatenate them with additional
    sensor data before final classification.

    Architecture:
    - Conv2d (3->8 channels, 5x5 kernel, stride=2) + ReLU + MaxPool2d (2x2)
    - Conv2d (8->16 channels, 5x5 kernel, stride=1, padding=2) + ReLU + MaxPool2d (2x2)
    - Flatten + Concatenate with relative position
    - Linear (2689->64) + ReLU
    - Linear (64->1) output

    Expected input image size: (61, 197) which gets processed to (16, 7, 24) after convolutions.
    """

    def __init__(self, num_classes):
        """Initialize the SimpleNetClassification25 model.

        Sets up all layers including convolutional layers, pooling, and fully connected layers.
        The input size calculation assumes input images of size (61, 197).
        """
        super(SimpleNetClassification25, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        # Adjust input size for (61,197) input to match NvidiaModel: 16*7*24 + 1 = 2689
        self.fc1 = nn.Linear(16 * 7 * 24 + 1, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).
                Expected input size is (batch_size, 3, 61, 197).
            relative_position (torch.Tensor): Relative position sensor data of shape
                (batch_size,) or (batch_size, 1). This additional sensor input is
                concatenated with the flattened convolutional features.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1). These are raw
                output values that can be used for regression or passed through
                a sigmoid for binary classification.

        Note:
            The network expects input images of size (61, 197). After the first
            conv+pool operation, the spatial dimensions become approximately (32, 32),
            and after the second conv+pool operation, they become (16, 16). The
            comment dimensions may not be accurate for all input sizes.
        """
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 8, ~15, ~49] for (61,197) input
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 16, 7, 24] for (61,197) input
        x = self.flatten(x)

        # Prepare additional inputs
        relative_position = relative_position.view(-1, 1)

        # Concatenate flattened conv output with additional sensor inputs
        x = torch.cat([x, relative_position], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits

        return x
