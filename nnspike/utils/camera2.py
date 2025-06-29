import cv2
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def calculate_red_area(image_path_or_array) -> float:
    """
    Calculate the area of red regions in an image using HSV color space.
    
    Args:
        image_path_or_array: Either a file path (string) or numpy array of the image
        
    Returns:
        float: Area of red regions in pixels
    """
    # Load image if path is provided, otherwise use the array directly
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array.copy()
    
    if image is None:
        return 0.0
    
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    # Red color wraps around in HSV, so we need two ranges
    # Lower red range (0-10)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    
    # Upper red range (170-180)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate the area (number of white pixels in the mask)
    red_area = cv2.countNonZero(red_mask)
    
    return float(red_area)


def visualize_red_detection(image_path_or_array, save_path=None) -> None:
    """
    Visualize the detected red areas in an image.
    
    Args:
        image_path_or_array: Either a file path (string) or numpy array of the image
        save_path: Optional path to save the visualization
    """
    # Load image if path is provided, otherwise use the array directly
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array.copy()
    
    if image is None:
        print("Error: Could not load image")
        return
    
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create visualization
    # Convert original image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create highlighted version where red areas are more prominent
    highlighted = image_rgb.copy()
    highlighted[red_mask > 0] = [255, 0, 0]  # Make detected red areas bright red
    
    # Create overlay version
    overlay = image_rgb.copy()
    overlay[red_mask > 0] = [255, 255, 0]  # Yellow overlay on detected areas
    result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    
    # Calculate area
    red_area = cv2.countNonZero(red_mask)
    
    # Create subplot visualization
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Red mask
    plt.subplot(2, 2, 2)
    plt.imshow(red_mask, cmap='gray')
    plt.title(f'Red Detection Mask\nArea: {red_area} pixels')
    plt.axis('off')
    
    # Highlighted red areas
    plt.subplot(2, 2, 3)
    plt.imshow(highlighted)
    plt.title('Detected Red Areas (Highlighted)')
    plt.axis('off')
    
    # Overlay visualization
    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.title('Red Areas with Yellow Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    print(f"Total red area detected: {red_area} pixels")