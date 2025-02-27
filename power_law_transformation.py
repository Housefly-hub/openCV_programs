import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_raw = cv.imread('./images/Rose.jpg')  

if img_raw is None:
    print("Image not found")
else:
    # Convert from BGR to RGB
    img = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)  

    # Normalize image (scale to 0-1)
    img_scaled = img / 255.0  

    # Apply power-law transformation
    gamma = 0.3  # Change this to experiment with different values
    img_pow = np.power(img_scaled, gamma) * 255  # Transform & rescale
    # Convert back to uint8
    img_pow = np.array(img_pow, dtype=np.uint8)

    #combining the original image and power-law transformed image
    combine_img = np.hstack((img,img_pow))

    '''Uncomment the below line to save the combined image of original image and power-law transformed image in the images folder'''
    # cv.imwrite("images/power_law_transformed_image.jpg",np.hstack((img_raw,cv.cvtColor(img_pow,cv.COLOR_RGB2BGR))))

    # Display original image
    plt.imshow(combine_img)
    plt.title("Original_image | Power-law_transformed_image")
    plt.axis("off")
    plt.show()
