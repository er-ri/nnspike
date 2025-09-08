import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        mode_weight: float = 1.0,
        control_weight: float = 30.0,
        control_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.mode_weight = mode_weight
        self.control_weight = control_weight
        self.control_scale = control_scale
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()  # or nn.SmoothL1Loss()

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode_output, control_output = outputs
        mode_target, control_target = targets

        mode_loss = self.classification_loss(mode_output, mode_target)
        # Scale only for loss calculation, keep outputs normalized
        scaled_control_loss = self.regression_loss(
            control_output * self.control_scale, control_target * self.control_scale
        )

        total_loss = (
            self.mode_weight * mode_loss + self.control_weight * scaled_control_loss
        )

        return total_loss, mode_loss, scaled_control_loss
