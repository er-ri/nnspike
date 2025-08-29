"""
This module defines custom PyTorch Datasets for driving records, including image preprocessing and augmentation.

Modules:
    - cv2: OpenCV library for image processing.
    - torch: PyTorch library for tensor operations and neural networks.
    - albumentations as A: Albumentations library for image augmentations.
    - torchvision.transforms as transforms: PyTorch's torchvision library for common image transformations.
    - PIL.Image: Python Imaging Library for image manipulation.
    - numpy as np: NumPy library for numerical operations.
    - torch.utils.data.Dataset: Base class for all datasets in PyTorch.
    - nnspike.utils.normalize_image: Custom function for image normalization.

Constants:
    - transformA (albumentations.ReplayCompose): Augmentation pipeline with random brightness/contrast adjustments and RGB shifts.
    - transform_flip (albumentations.Compose): Augmentation pipeline for horizontal flipping of images.

Classes:
    - NvidiaDataset(Dataset): Custom dataset class for loading and preprocessing driving record data for Nvidia model.
    - MobileNetV2Dataset(Dataset): Custom dataset class for loading and preprocessing driving record data for MobileNetV2 model.

NvidiaDataset Class:
    Methods:
        - __init__(self, inputs, offset_xs, roi, train_course):
            Initializes the dataset with input image paths, corresponding labels, region of interest, and training course.
        
        - __len__(self):
            Returns the number of samples in the dataset.
        
        - __getitem__(self, idx):
            Retrieves and processes the sample at the given index. This includes:
                - Reading the image from the file path using OpenCV.
                - Extracting the region of interest (ROI) from the image.
                - Applying image augmentations such as brightness/contrast adjustments and RGB shifts.
                - Normalizing the ROI.
                - Converting the ROI to a PyTorch tensor.
                - Adjusting the steering angle label if a horizontal flip was applied.
                - Scaling the interval and label values.
                - Returning the processed ROI and interval as input features, and the label as the target.

MobileNetV2Dataset Class:
    Methods:
        - __init__(self, inputs, offset_xs, roi, train_course):
            Initializes the dataset with input image paths, corresponding labels, region of interest, and training course.
        
        - __len__(self):
            Returns the number of samples in the dataset.
        
        - __getitem__(self, idx):
            Retrieves and processes the sample at the given index. This includes:
                - Reading the image from the file path using PIL.
                - Extracting the region of interest (ROI) from the image.
                - Applying image augmentations such as brightness/contrast adjustments and RGB shifts.
                - Normalizing the ROI.
                - Converting the ROI to a PyTorch tensor.
                - Adjusting the steering angle label if a horizontal flip was applied.
                - Scaling the label values.
                - Returning the processed ROI as input feature, and the label as the target.

Usage Example:
    nvidia_dataset = NvidiaDataset(inputs=[('path/to/image.png', 1, 'course1')], offset_xs=[50], roi=(0, 0, 200, 200), train_course='course1')
    mobilenetv2_dataset = MobileNetV2Dataset(inputs=[('path/to/image.png', 'course1')], offset_xs=[50], roi=(0, 0, 200, 200), train_course='course1')
    
    nvidia_dataloader = torch.utils.data.DataLoader(nvidia_dataset, batch_size=4, shuffle=True)
    mobilenetv2_dataloader = torch.utils.data.DataLoader(mobilenetv2_dataset, batch_size=4, shuffle=True)
    
    for (roi_area, interval), label in nvidia_dataloader:
        # Training loop here
    
    for roi_area, label in mobilenetv2_dataloader:
        # Training loop here

Note:
    - The `normalize_image` function should be defined in the `nnspike.utils` module.
    - The `transformA` object applies random brightness/contrast adjustments and RGB shifts to the images.
    - The `transform_flip` object applies horizontal flips to the images.
"""

import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from nnspike.utils import normalize_image

transformA = A.ReplayCompose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    ]
)

transform_flip = A.Compose(
    [A.HorizontalFlip(p=1)],
)


class NvidiaDataset(Dataset):

    preprocess = transforms.ToTensor()

    def __init__(self, inputs, offset_xs, roi, train_course):
        self.inputs = inputs
        self.offset_xs = offset_xs
        self.roi = roi
        self.train_course = train_course

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image_path = self.inputs[idx][0]
        image = cv2.imread(image_path)
        roi_area = image[self.roi[1] : self.roi[3], self.roi[0] : self.roi[2]]
        interval = self.inputs[idx][1]
        course = self.inputs[idx][2]

        label = self.offset_xs[idx] - self.roi[0]

        roi_area = transformA(image=roi_area)["image"]

        # Horizontal flip the image if the course is not equal to the training course
        if course != self.train_course:
            roi_area = transform_flip(image=roi_area)["image"]
            label = (self.roi[2] - self.roi[0]) - label

        roi_area = normalize_image(image=roi_area)
        roi_area = self.preprocess(roi_area)  # Convert to pytorch tensor
        roi_area = roi_area.to(torch.float32)  # `Conv2d` supports up to `float32`

        interval = torch.tensor(interval, dtype=torch.float32)

        label = (label) / (self.roi[2] - self.roi[0])
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

        return tuple([roi_area, interval]), label


class MobileNetV2Dataset(Dataset):

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, inputs, offset_xs, roi, train_course):
        self.inputs = inputs
        self.offset_xs = offset_xs
        self.roi = roi
        self.train_course = train_course

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image_path = self.inputs[idx][0]
        image = Image.open(image_path)
        roi_area = image.crop((self.roi[0], self.roi[1], self.roi[2], self.roi[3]))
        roi_area = np.asarray(roi_area)
        course = self.inputs[idx][1]

        label = self.offset_xs[idx]

        transformed_roi_area = transformA(image=roi_area)
        roi_area = transformed_roi_area["image"]

        # Horizontal flip the image if the course is not equal to the training course
        if course != self.train_course:
            roi_area = transform_flip(image=roi_area)["image"]
            label = (self.roi[2] - self.roi[0]) - label

        roi_area = Image.fromarray(np.uint8(roi_area))
        roi_area = self.preprocess(roi_area)
        roi_area = roi_area.to(torch.float32)

        label = label / (self.roi[2] - self.roi[0])

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

        return roi_area, label
