import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot


def resize_image(image, max_size=500):
    """
    Resizes the image while maintaining the original aspect ratio.

    Parameters:
        image (numpy.ndarray): Image to be resized.
        max_size (int): Maximum allowed width or height for the resized image.

    Returns:
        numpy.ndarray: Resized image with preserved aspect ratio.
    """
    # Get original dimensions of the image
    h, w = image.shape[:2]  
    
    # Compute the scale factor that ensures the largest dimension fits within max_size
    scale = min(max_size / w, max_size / h)  
    
    # Calculate the new dimensions using the computed scale factor
    new_w, new_h = int(w * scale), int(h * scale)  
    
    # Resize the image using INTER_AREA interpolation (good for downscaling)
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
