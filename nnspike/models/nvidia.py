import torch
import torch.nn as nn


class NvidiaModelRegression(nn.Module):
    """A neural network model based on the NVIDIA architecture for end-to-end learning of self-driving cars.

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
        fc1 (nn.Linear): First fully connected layer with input size adjusted to include sensor inputs.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        mode_classifier (nn.Linear): Output layer for behavior mode classification (4 modes).
        self_driving_head (nn.Linear): Output layer for self-driving control.
        elu (nn.ELU): Exponential Linear Unit activation function applied after each layer except the final output layer.
        softmax (nn.Softmax): Softmax activation for mode classification.

    Methods:
        forward(x, left_x, right_x, relative_position):
            Defines the forward pass of the model. Takes an image tensor `x` and additional sensor inputs,
            processes them through the network, and returns two output tensors: mode classification and control.

    Args:
        x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).
        left_x (torch.Tensor): Left sensor input tensor of shape (batch_size, 1).
        right_x (torch.Tensor): Right sensor input tensor of shape (batch_size, 1).
        relative_position (torch.Tensor): Relative position tensor of shape (batch_size, 1).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - mode_output: Softmax probabilities for robot behavior modes (batch_size, 4)
              [left_x following, right_x following, obstacle avoidance, self driving]
            - control_output: Control tensor for self-driving mode (batch_size, 1)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            64 * 1 * 18 + 1, 100
        )  # Adjust input size to include relative_position
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

        self.elu = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = self.flatten(x)

        # Prepare additional inputs
        relative_position = relative_position.view(-1, 1)

        # Concatenate flattened conv output with additional sensor inputs
        x = torch.cat([x, relative_position], dim=1)

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)

        return x


class NvidiaModelMultiTask(nn.Module):
    """A neural network model based on the NVIDIA architecture for end-to-end learning of self-driving cars.

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
        fc1 (nn.Linear): First fully connected layer with input size adjusted to include sensor inputs.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        mode_classifier (nn.Linear): Output layer for behavior mode classification (4 modes).
        self_driving_head (nn.Linear): Output layer for self-driving control.
        elu (nn.ELU): Exponential Linear Unit activation function applied after each layer except the final output layer.
        softmax (nn.Softmax): Softmax activation for mode classification.

    Methods:
        forward(x, left_x, right_x, relative_position):
            Defines the forward pass of the model. Takes an image tensor `x` and additional sensor inputs,
            processes them through the network, and returns two output tensors: mode classification and control.

    Args:
        x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).
        left_x (torch.Tensor): Left sensor input tensor of shape (batch_size, 1).
        right_x (torch.Tensor): Right sensor input tensor of shape (batch_size, 1).
        relative_position (torch.Tensor): Relative position tensor of shape (batch_size, 1).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - mode_output: Softmax probabilities for robot behavior modes (batch_size, 4)
              [left_x following, right_x following, obstacle avoidance, self driving]
            - control_output: Control tensor for self-driving mode (batch_size, 1)
    """

    def __init__(self, num_modes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            64 * 1 * 18 + 1, 100
        )  # Adjust input size to include left_x, right_x, and relative_position
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        # Mode classification head (4 modes: left_x following, right_x following, obstacle avoidance, self driving)
        self.mode_head = nn.Linear(10, num_modes)

        # Self-driving control head
        self.self_driving_head = nn.Linear(10, 1)

        self.elu = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        relative_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = self.flatten(x)

        # Prepare additional inputs
        relative_position = relative_position.view(-1, 1)

        # Concatenate flattened conv output with additional sensor inputs
        x = torch.cat([x, relative_position], dim=1)

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))

        # Mode classification output (softmax for behavior mode)
        mode_output = self.mode_head(x)

        # Self-driving control output
        control_output = self.self_driving_head(x)

        return mode_output, control_output
