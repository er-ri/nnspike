import numpy as np
import torch
import torchvision.transforms as transforms

from nnspike.constants import ROI_CNN
from nnspike.utils import normalize_image

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest

transform = transforms.ToTensor()


def process_image(
    image: np.ndarray, roi: tuple[int, int, int, int], device: torch.device
) -> torch.Tensor:
    x1, y1, x2, y2 = roi

    roi_area = image[y1:y2, x1:x2]
    roi_area = normalize_image(image=roi_area)
    tensor_roi_area = transform(roi_area)
    tensor_roi_area = tensor_roi_area.to(torch.float32)
    tensor_roi_area = tensor_roi_area.unsqueeze(0)
    tensor_roi_area = tensor_roi_area.to(device)

    return tensor_roi_area
