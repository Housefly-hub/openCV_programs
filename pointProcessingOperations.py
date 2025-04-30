import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('images/Rose.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Make sure 'input.jpg' exists in your working directory.")

# 1. Negative Transform
negative_img = 255 - img

# 2. Log Transform
img_float = img.astype(np.float32)
log_img = cv2.normalize(np.log1p(img_float), None, 0, 255, cv2.NORM_MINMAX)
log_img = log_img.astype(np.uint8)

# 3. Power Law (Gamma) Transform
gamma = 0.5  # You can change gamma for different effects
power_img = cv2.normalize(np.power(img_float / 255.0, gamma), None, 0, 255, cv2.NORM_MINMAX)
power_img = power_img.astype(np.uint8)

# Plot all images
titles = ['Original Image', 'Negative Transform', 'Log Transform', 'Power Law Transform (γ=0.5)']
images = [img, negative_img, log_img, power_img]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('images/image_transforms.png', dpi=300)
plt.show()
