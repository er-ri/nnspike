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


def get_dominant_color(image):
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    dominant_color = palette[0]

    # Determine if the dominant color is closer to blue or red
    blue_distance = np.linalg.norm(dominant_color - np.array([255, 0, 0]))
    red_distance = np.linalg.norm(dominant_color - np.array([0, 0, 255]))

    if blue_distance < red_distance:
        return "blue"
    else:
        return "red"
