"""Checkpoint management utilities for model training.

This module provides functions for saving and loading PyTorch model checkpoints
during training. It handles the persistence of model weights, optimizer states,
training epochs, and loss values to enable training resumption and model deployment.

Typical usage example:

    from nnspike.utils.checkpoint import save_checkpoint, load_checkpoint

    # Save a checkpoint during training
    save_checkpoint(model, optimizer, epoch=10, loss=0.25, save_path='checkpoint.pth')

    # Load a checkpoint to resume training
    epoch, loss = load_checkpoint(model, 'checkpoint.pth', optimizer=optimizer, device='cuda')
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
) -> None:
    """Save model checkpoint.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer state to save.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        save_path (str): Path to save the checkpoint file.

    Returns:
        None

    Example:
        >>> model = UNet(n_channels=3, n_classes=2)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scaler.fit(training_data)
        >>> save_checkpoint(
        ...     model,
        ...     optimizer,
        ...     epoch=10,
        ...     loss=0.25,
        ...     save_path="checkpoint.pth",
        ... )
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: optim.Optimizer | None = None,
    device: str = "cpu",
) -> tuple[int, float]:
    """Load model checkpoint.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (optim.Optimizer | None, optional): Optional optimizer to load state into.
            Defaults to None.
        device (str, optional): Device to load the model on ('cpu', 'cuda', etc.).
            Defaults to "cpu".

    Returns:
        tuple[int, float, object | None]: Tuple containing:
            - epoch: The epoch number when checkpoint was saved.
            - loss: The loss value at that epoch.

    Example:
        >>> model = UNet(n_channels=3, n_classes=2)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> epoch, loss = load_checkpoint(
        ...     model, "checkpoint.pth", optimizer=optimizer, device="cuda"
        ... )
        >>> print(f"Resumed from epoch {epoch} with loss {loss:.4f}")
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return epoch, loss
