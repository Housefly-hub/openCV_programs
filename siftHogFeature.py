import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Read the input image
image = cv2.imread('images/Childrens.jpg')  # Replace with your image path

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- SIFT Feature Detection ---
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_image, None)
sift_image = cv2.drawKeypoints(image, keypoints, None)

# --- HOG Feature Detection using skimage ---
fd, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# --- Plotting ---
plt.figure(figsize=(18, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# SIFT Features
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
plt.title("SIFT Features")
plt.axis('off')

# HOG Features
plt.subplot(1, 3, 3)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title("HOG Features")
plt.axis('off')

# Save the final output
plt.tight_layout()
plt.savefig('images/sift_hog_original_output.jpg')
plt.show()
