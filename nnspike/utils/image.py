"""This module provides utility functions for image and video processing using OpenCV and NumPy.

Functions:
    normalize_image(image: np.ndarray) -> np.ndarray:
        Normalize an input image by converting its color space, applying Gaussian blur,
        resizing, and scaling pixel values.

    draw_driving_info(image: np.ndarray, info: dict, roi: tuple[int, int, int, int]) -> np.ndarray:
        Draws driving information on an image by overlaying a tracing point, a region of interest (ROI)
        rectangle, and various text annotations based on the provided info dictionary.

    extract_video_frames(video_path: str, frame_path: str) -> None:
        Extracts frames from a video file and saves them as individual image files in the specified directory.
"""

import cv2
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an input image by converting its color space, applying Gaussian blur, resizing, and scaling pixel values.

    This function performs the following steps:
    1. Converts the image from RGB to YUV color space.
    2. Applies a Gaussian blur with a kernel size of 5x5.
    3. Resizes the image to dimensions 200x66.
    4. Scales the pixel values to the range [0, 1].

    Args:
        image (np.ndarray): Input image in RGB format as a NumPy array.

    Returns:
        np.ndarray: Normalized image as a NumPy array.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255

    return image


def draw_driving_info(
    image: np.ndarray, info: dict, roi: tuple[int, int, int, int]
) -> np.ndarray:
    """Draws driving information on an image.

    This function overlays driving-related information onto a given image. It draws a tracing point,
    a region of interest (ROI) rectangle, and various text annotations based on the provided info dictionary.

    Args:
        image (np.ndarray): The input image on which to draw the information.
        info (dict): A dictionary containing the driving information to be displayed.
            Expected keys are:
                - "trace_x" (int or str): The x-coordinate for the tracing point.
                - "trace_y" (int or str): The y-coordinate for the tracing point.
                - "text" (dict): A dictionary of text annotations where keys are the labels and values are the corresponding data.
        roi (tuple[int, int, int, int]): A tuple defining the region of interest in the format (x1, y1, x2, y2).

    Returns:
        np.ndarray: The image with the overlaid driving information.
    """
    target_x, offset_y = int(info["target_x"]), int(info["offset_y"])
    x1, y1, x2, y2 = roi

    # Draw a cross line in (target_x, offset_y)
    if target_x != 0:
        image = cv2.circle(
            image, (target_x, offset_y), 5, (255, 255, 0), -1
        )  # Tracing point

        cv2.line(
            image,
            (target_x - 100, offset_y),
            (target_x + 100, offset_y),
            (255, 255, 0),
            1,
        )
        cv2.line(
            image,
            (target_x, offset_y - 100),
            (target_x, offset_y + 100),
            (255, 255, 0),
            1,
        )

    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # ROI

    for index, key in enumerate(info["text"]):
        value = info["text"][key]
        text = f"{value:.2f}" if type(value) is float else value

        image = cv2.putText(
            image,
            f"{key} : {text}",
            (50, 20 + index * 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),  # Font color
            1,
            cv2.LINE_4,
        )

    return image
