import torch
import torchvision.transforms as transforms

from nnspike.constants import ROI_CNN
from nnspike.utils import normalize_image

# User defined constants
x1, y1, x2, y2 = ROI_CNN  # Region of Interest

transform = transforms.ToTensor()


def process_image(image, roi, device):
    x1, y1, x2, y2 = roi

    roi_area = image[y1:y2, x1:x2]
    roi_area = normalize_image(image=roi_area)
    roi_area = transform(roi_area)
    roi_area = roi_area.to(torch.float32)
    roi_area = roi_area.unsqueeze(0)
    roi_area = roi_area.to(device)

    return roi_area


def model_inference(model, roi_area, tensor_relative_position):
    """
    Perform inference using the model on the given ROI area and relative position.

    Args:
        model: The trained model for inference.
        roi_area: The region of interest image tensor.
        relative_position: The relative position tensor.

    Returns:
        The model's output predictions.
    """
    with torch.no_grad():
        outputs = model(roi_area, tensor_relative_position)

    prob, mode = torch.max(outputs[0], dim=1)
    prob_value = round(prob[0].item(), 2)
    mode_value = mode.item()  # Convert to Python integer
    predicted_x = x1 + (outputs[1][0][0] * (x2 - x1)).detach().item()

    return predicted_x, (mode_value, prob_value)
