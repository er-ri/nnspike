import torch
import torch.nn as nn

from nnspike.constants import NUM_MODES


class NvidiaModel(nn.Module):
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

    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 1 * 18 + 1, 100)  # Adjust input size to include left_x, right_x, and relative_position
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        # Mode classification head (4 modes: left_x following, right_x following, obstacle avoidance, self driving)
        self.mode_classifier = nn.Linear(10, NUM_MODES)

        # Self-driving control head
        self.self_driving_head = nn.Linear(10, 1)

        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x,
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
        mode_output = self.softmax(self.mode_classifier(x))

        # Self-driving control output
        control_output = self.self_driving_head(x)

        return mode_output, control_output


class MultiTaskLoss(nn.Module):

    def __init__(self, mode_weight=1.0, control_weight=30.0, control_scale=10.0):
        super(MultiTaskLoss, self).__init__()
        self.mode_weight = mode_weight
        self.control_weight = control_weight
        self.control_scale = control_scale
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()  # or nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        mode_output, control_output = outputs
        mode_target, control_target = targets

        mode_loss = self.classification_loss(mode_output, mode_target)
        # Scale only for loss calculation, keep outputs normalized
        scaled_control_loss = self.regression_loss(control_output * self.control_scale, control_target * self.control_scale)

<<<<<<< HEAD
        # print(f"Control target range: min={control_target.min():.6f}, max={control_target.max():.6f}")
        # print(f"Control target std: {control_target.std():.6f}")

=======
>>>>>>> wip/teamwork-li
        total_loss = self.mode_weight * mode_loss + self.control_weight * scaled_control_loss

        return total_loss, mode_loss, scaled_control_loss
