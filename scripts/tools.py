import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import cv2
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import process_image


def evaluate_model_for_dataset(df, model, device, roi) -> pd.DataFrame:
    x1, _, x2, _ = roi

    diff = list()

    for _, row in df.iterrows():
        if not pd.isna(row["adjusted_x"]):
            offset_x = row["adjusted_x"]
        elif not pd.isna(row["predicted_x"]):
            offset_x = row["predicted_x"]
        else:
            offset_x = row["mx"]
        row["offset_x"] = offset_x

        image = cv2.imread(row["image_path"])

        roi_area = process_image(image.copy(), device, roi)

        with torch.no_grad():
            output = model(roi_area)

        predicted_x = x1 + (output[0][0] * (x2 - x1)).detach().item()
        diff = offset_x - predicted_x
        diff.append(diff)

    df["diff"] = diff

    return df


def view_data_distribution(df: pd.DataFrame) -> None:
    sns.set_style("darkgrid", {"grid.color": ".5", "grid.linestyle": ":"})

    plt.figure(figsize=(15, 6))

    # Plot the 'offset_x' distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df["offset_x"], bins=30, color="steelblue", edgecolor=None)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of 'offset_x'")

    # Plot the 'line_type' distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df["line_type"], bins=20, color="forestgreen", edgecolor=None)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of 'line_type'")

    # Display the combined plot
    plt.tight_layout()
    plt.show()
