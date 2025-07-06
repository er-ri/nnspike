import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from nnspike.utils import normalize_image
from nnspike.models import NvidiaModel

transform = transforms.ToTensor()


def load_and_prepare_model(model_path, device):
    model = NvidiaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def process_image(image, device, roi):
    x1, y1, x2, y2 = roi

    roi_area = image[y1:y2, x1:x2]
    roi_area = normalize_image(image=roi_area)
    roi_area = transform(roi_area)
    roi_area = roi_area.to(torch.float32)
    roi_area = roi_area.unsqueeze(0)
    roi_area = roi_area.to(device)

    return roi_area
