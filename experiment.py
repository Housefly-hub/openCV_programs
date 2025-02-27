import cv2 as cv
import numpy as np
from image_mean_filter import resize_image 

image = cv.imread("./images/Avatar_Boy.png",cv.IMREAD_GRAYSCALE)
print(image)
print("\n\n")

image_resized = resize_image(image,10)
print(image_resized)
print("\n\n")

padded_image = cv.copyMakeBorder(image_resized,1,1,1,1,cv.BORDER_CONSTANT)
print(padded_image)
print("\n\n")

filtered_image = np.zeros_like(image_resized)


for y in range(image_resized.shape[0]):
    for x in range(image_resized.shape[1]):
        window = padded_image[y:y+3, x:x+3]
        print(window)
        print(x,y,end='')
        print('\n')
        mean_value = np.mean(window)
        filtered_image[y,x] = mean_value



