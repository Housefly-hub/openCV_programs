import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('images/Butterfly.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Make sure 'input.jpg' is in your working directory.")

### ----- Intensity Level Slicing -----
# Slicing ranges
slice1 = cv2.inRange(img, 100, 150)     # pixels between 100 and 150
slice2 = cv2.inRange(img, 50, 100)      # pixels between 50 and 100
slice3 = cv2.inRange(img, 150, 200)     # pixels between 150 and 200

# Convert to full 255 binary for visualization
slice1_img = cv2.bitwise_and(img, img, mask=slice1)
slice2_img = cv2.bitwise_and(img, img, mask=slice2)
slice3_img = cv2.bitwise_and(img, img, mask=slice3)

# Plot Intensity Level Slicing
plt.figure(figsize=(10, 4))
titles = ['Original Image', 'Gray Level Slicing - 1 (100–150)', 'Gray Level Slicing - 2 (50–100)', 'Gray Level Slicing - 3 (150–200)']
images = [img, slice1_img, slice2_img, slice3_img]

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('images/intensity_level_slicing.png', dpi=300)
plt.show()

### ----- Bit Plane Slicing -----
bit_planes = [(img >> i) & 1 for i in range(8)]
bit_planes_imgs = [plane * 255 for plane in bit_planes]

# Plot Bit Plane Slicing
plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

for i in range(1, 8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(bit_planes_imgs[i], cmap='gray')
    plt.title(f"Bit Plane {i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('images/bit_plane_slicing.png', dpi=300)
plt.show()
