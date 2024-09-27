import torch.nn as nn
from torchvision import models


class MobileNetV2Regression(nn.Module):
    def __init__(self, num_outputs=1):
        super(MobileNetV2Regression, self).__init__()

        # Load the pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(
            weights=models.mobilenetv2.MobileNet_V2_Weights.IMAGENET1K_V2
        )

        # Replace the classifier with a new fully connected layer for regression
        self.mobilenet_v2.classifier[1] = nn.Linear(
            self.mobilenet_v2.last_channel, num_outputs
        )

    def forward(self, x):
        return self.mobilenet_v2(x)
