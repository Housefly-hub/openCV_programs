import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter

# Read image in grayscale
image = cv2.imread("images/Elephant.jpg", cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    raise FileNotFoundError("Image not found. Make sure 'your_image.jpg' exists in the same folder.")

# Define kernel sizes
kernel_sizes = [3, 5, 7]

# Define filters
def box_filter(img, k):
    kernel = np.ones((k, k), np.float32) / (k * k)
    return cv2.filter2D(img, -1, kernel)

def tent_filter(img, k):
    tent_kernel = np.convolve(np.ones(k), np.ones(k))
    tent_kernel = tent_kernel / tent_kernel.sum()
    kernel_2d = np.outer(tent_kernel, tent_kernel)
    return cv2.filter2D(img, -1, kernel_2d)

def gaussian_filter(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)

def min_filter(img, k):
    return minimum_filter(img, size=(k, k))

def max_filter(img, k):
    return maximum_filter(img, size=(k, k))

filters = {
    "Box Filter": box_filter,
    "Tent Filter": tent_filter,
    "Min Filter": min_filter,
    "Max Filter": max_filter,
    "Gaussian Filter": gaussian_filter
}

# Create subplot grid: 6 rows (1 input + 5 filters), 4 columns (1 label + 3 kernel sizes)
fig, axs = plt.subplots(6, 4, figsize=(16, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# Set top row titles for kernel sizes (columns 1,2,3)
axs[0, 0].text(0.5, 0.5, 'Input Image', fontsize=12, ha='center', va='center')
axs[0, 0].axis('off')
for j, k in enumerate(kernel_sizes):
    axs[0, j + 1].imshow(image, cmap='gray')
    axs[0, j + 1].set_title(f'{k}x{k}')
    axs[0, j + 1].axis('off')

# Plot filtered outputs
for i, (name, func) in enumerate(filters.items()):
    axs[i + 1, 0].text(0.5, 0.5, name, fontsize=12, ha='center', va='center')
    axs[i + 1, 0].axis('off')
    for j, k in enumerate(kernel_sizes):
        filtered = func(image, k)
        axs[i + 1, j + 1].imshow(filtered, cmap='gray')
        axs[i + 1, j + 1].axis('off')

plt.suptitle("Various Image filters", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("images/Various_Image_Filters.png")
plt.show()
