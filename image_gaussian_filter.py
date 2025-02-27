''' Convolution using scipy library'''

# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plot
# from image_mean_filter import resize_image
# from scipy.ndimage import convolve

# def gaussian_kernel(size,size_y = None):
#     size = int(size)
#     if not size_y:
#         size_y = size
#     else:
#         size_y = int(size_y)
    
#     x,y = np.mgrid[-size:size+1, -size_y:size_y+1]
#     g = np.exp(-(x**2/float(size) + y**2/float(size_y)))
#     return g/g.sum()

# # Make the Gaussian kernel by calling the function
# kernel_size = 7

# gaussian_kernel_array = gaussian_kernel(kernel_size)

# #load image in grayscale
# image = cv.imread('images/Girl.png',cv.IMREAD_GRAYSCALE)

# filtered_image = (convolve(image,gaussian_kernel_array))
# filtered_image = resize_image(filtered_image)

# cv.imshow('Original Image', image)
# cv.imshow('Gaussian Filtered Image', filtered_image)
# cv.waitKey(0)
# cv.destroyAllWindows()


''' Manual Convolution '''

# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plot

# def gaussian_kernel(size, sigma):
#     """
#     Generates a Gaussian kernel.

#     Parameters:
#     - size: Size of the kernel (must be odd).
#     - sigma: Standard deviation of the Gaussian function.

#     Returns:
#     - Normalized Gaussian kernel.
#     """
#     k = size // 2  # Kernel radius
#     x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
#     gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
#     return gaussian / np.sum(gaussian)  # Normalize the kernel

# def apply_gaussian_filter(image, kernel):
#     """
#     Applies Gaussian smoothing using manual convolution.

#     Parameters:
#     - image: Grayscale image.
#     - kernel: Gaussian kernel.

#     Returns:
#     - Filtered image.
#     """
#     h, w = image.shape
#     k_size = kernel.shape[0]
#     pad = k_size // 2  # Padding size

#     # Pad the image with zero-padding (to handle border pixels)
#     padded_image = np.pad(image, pad, mode='constant', constant_values=0)

#     # Create an empty image to store the filtered result
#     filtered_image = np.zeros_like(image)

#     # Perform convolution
#     for i in range(h):
#         for j in range(w):
#             # Extract the region of interest
#             region = padded_image[i:i+k_size, j:j+k_size]
#             # Apply element-wise multiplication and sum up
#             filtered_image[i, j] = np.sum(region * kernel)

#     return filtered_image

# # Load the image in grayscale
# image = cv.imread('images/Girl.png', cv.IMREAD_GRAYSCALE)

# if image is None:
#     print("Error: Could not load image.")
#     exit()

# # Define kernel size and standard deviation
# kernel_size = 7  # Must be odd
# sigma = 1.0

# # Generate Gaussian kernel
# gaussian_k = gaussian_kernel(kernel_size, sigma)

# # Apply Gaussian smoothing using manual convolution
# smoothed_image = apply_gaussian_filter(image, gaussian_k)

# # Display the results
# plt.figure(figsize=(10, 5))

# plot.subplot(1, 2, 1)
# plot.imshow(image, cmap='gray')
# plot.title("Original Image")
# plot.axis("off")

# plot.subplot(1, 2, 2)
# plot.imshow(smoothed_image, cmap='gray')
# plot.title("Gaussian Smoothed Image (Manual)")
# plot.axis("off")

# plot.show()






''' Convolution using the inbuilt openCV function for gaussian blur GaussianBlur()
    Most efficient since its optimized at low level'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from snippet import resize_image

def apply_gaussian_filter(image, kernel_size, sigma):
    """
    Applies Gaussian smoothing to the input image.
    
    Parameters:
    - image: Grayscale image
    - kernel_size: Tuple (width, height) of the Gaussian kernel (must be odd)
    - sigma: Standard deviation for Gaussian distribution

    Returns:
    - Filtered image
    """
    return cv.GaussianBlur(image, kernel_size, sigma)

# Load the image in grayscale
image = cv.imread('images/Girl.png', cv.IMREAD_GRAYSCALE)

# Check if image loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Define Gaussian kernel size and standard deviation
kernel_size = (7,7)  # Must be an odd number (e.g., 3x3, 5x5, 7x7)
sigma = 1.0  # Standard deviation for Gaussian distribution

# Apply Gaussian filter
smoothed_image = apply_gaussian_filter(image, kernel_size, sigma)
combine_img = np.hstack((image,smoothed_image))
combine_img = resize_image(combine_img)

#save the combined image of original image and gaussian filtered image in the image folder
cv.imwrite("images/gaussianfiltered_image.jpg",combine_img)

#Display image

plot.imshow(combine_img, cmap='gray')
plot.title("Original_image | Gaussian_filtered_image")
plot.axis("off")
plot.show()














