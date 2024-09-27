import torch
import torch.nn as nn


class NvidiaModel(nn.Module):
    """
    A neural network model inspired by the NVIDIA architecture for end-to-end learning of self-driving cars.

    This model consists of a series of convolutional layers followed by fully connected layers. The input to the
    model is an image and an additional stage parameter, which is concatenated with the flattened output of the
    convolutional layers before being passed through the fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        conv5 (nn.Conv2d): Fifth convolutional layer.
        flatten (nn.Flatten): Layer to flatten the output of the convolutional layers.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Fourth fully connected layer.
        elu (nn.ELU): Exponential Linear Unit activation function.

    Methods:
        forward(x):
            Defines the forward pass of the model. Takes an image tensor `x`,
            processes it through the network, and returns the output tensor.
    """

    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 1 * 18, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=1)
        self.elu = nn.ELU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))

        x = self.flatten(x)

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)

        return x


class NvidiaModelV2(nn.Module):
    """
    A neural network model based on the NVIDIA architecture for end-to-end learning of self-driving cars.

    This model consists of five convolutional layers followed by four fully connected layers. The ELU activation
    function is used after each layer except the final output layer. Additionally, an interval input is concatenated
    with the flattened output from the convolutional layers before being passed through the fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 3 input channels and 24 output channels.
        conv2 (nn.Conv2d): Second convolutional layer with 24 input channels and 36 output channels.
        conv3 (nn.Conv2d): Third convolutional layer with 36 input channels and 48 output channels.
        conv4 (nn.Conv2d): Fourth convolutional layer with 48 input channels and 64 output channels.
        conv5 (nn.Conv2d): Fifth convolutional layer with 64 input channels and 64 output channels.
        flatten (nn.Flatten): Layer to flatten the output from the convolutional layers.
        fc1 (nn.Linear): First fully connected layer with input size adjusted to include the interval.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Fourth fully connected layer producing the final output.
        elu (nn.ELU): Exponential Linear Unit activation function applied after each layer except the final output layer.

    Methods:
        forward(x, interval):
            Defines the forward pass of the model. Takes an image tensor `x` and an interval tensor `interval` as inputs,
            processes them through the network, and returns the output tensor.

    Args:
        x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).
        interval (torch.Tensor): Interval tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1).
    """

    def __init__(self):
        super(NvidiaModelV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            64 * 1 * 18 + 1, 100
        )  # Adjust input size to include interval
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.elu = nn.ELU()

    def forward(self, x, interval):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = self.flatten(x)

        interval = interval.view(-1, 1)
        x = torch.cat((x, interval), dim=1)

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)

        return x
