import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img_raw = cv.imread("./images/Rose.jpg")  # openCV reads the image in BGR format

# Check if the image was loaded correctly
if img_raw is None:
    print("Error: Image not found. Check the file path.")
else:
    # Convert BGR to RGB
    img = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

    # Apply negative transformation
    img_neg = cv.bitwise_not(img) # It is equivalent of Ixy = 225 - Ix0y0


    #combine the original image and negative_transformed image
    combine_img = np.hstack((img,img_neg))

    '''Uncomment below line to save the combined image in the images folder'''
    # cv.imwrite("images/negative_transformed_image.jpg",np.hstack((img_raw,cv.cvtColor(img_neg,cv.COLOR_RGB2BGR))))

    # Show original image
    plt.imshow(combine_img)
    plt.title("Original_image | Negative_transformed_image")
    plt.axis("off")  # Hide axis
    plt.show()
