import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('images/Strawberry.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for segmentation (example: red color range)
# For RGB
lower_rgb = np.array([150, 0, 0])
upper_rgb = np.array([255, 100, 100])
mask_rgb = cv2.inRange(image_rgb, lower_rgb, upper_rgb)
segmented_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_rgb)

# For HSV
lower_hsv = np.array([0, 100, 100])
upper_hsv = np.array([10, 255, 255])
mask_hsv = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
segmented_hsv = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_hsv)  # Display in RGB

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(segmented_rgb)
plt.title('RGB Segmentation')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(segmented_hsv)
plt.title('HSV Segmentation')
plt.axis('off')

plt.tight_layout()
plt.savefig('images/segmentation_rgb_hsv.png')
plt.show()
