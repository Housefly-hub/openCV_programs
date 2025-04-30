import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read input image in grayscale
img = cv2.imread('images/Kittens.jpg', cv2.IMREAD_GRAYSCALE)

# Prewitt Operator
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(img, -1, kernelx)
prewitt_y = cv2.filter2D(img, -1, kernely)
prewitt = cv2.add(prewitt_x, prewitt_y)

# Sobel Operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

# Canny Edge
canny = cv2.Canny(img, 100, 200)

# Plot results in a single row
plt.figure(figsize=(16, 4))

titles = ['Input Image', 'Prewitt Edge', 'Sobel Edge', 'Canny Edge']
images = [img, prewitt, sobel, canny]

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig("images/edge_detection.png")
plt.show()
