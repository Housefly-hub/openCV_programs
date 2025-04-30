import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretch(img, r1, s1, r2, s2):
    """Apply piecewise linear contrast stretching with control points."""
    result = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            if pixel < r1:
                result[i, j] = (s1 / r1) * pixel
            elif pixel < r2:
                result[i, j] = ((s2 - s1) / (r2 - r1)) * (pixel - r1) + s1
            else:
                result[i, j] = ((255 - s2) / (255 - r2)) * (pixel - r2) + s2
    return result

# Load grayscale image
img = cv2.imread('images/Dog.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Make sure 'input.jpg' exists in your working directory.")

# Apply contrast stretching with different control points
params = [
    (50, 0, 200, 255),
    (70, 30, 180, 220),
    (100, 50, 150, 200)
]

out1 = contrast_stretch(img, *params[0])
out2 = contrast_stretch(img, *params[1])
out3 = contrast_stretch(img, *params[2])

# Plot original and outputs with control point titles
titles = [
    'Original Image',
    f'Output 1\n(r1={params[0][0]}, s1={params[0][1]}, r2={params[0][2]}, s2={params[0][3]})',
    f'Output 2\n(r1={params[1][0]}, s1={params[1][1]}, r2={params[1][2]}, s2={params[1][3]})',
    f'Output 3\n(r1={params[2][0]}, s1={params[2][1]}, r2={params[2][2]}, s2={params[2][3]})'
]
images = [img, out1, out2, out3]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('images/contrast_stretching_outputs_labeled.png', dpi=300)
plt.show()
