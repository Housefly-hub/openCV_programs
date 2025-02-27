import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot

img_raw = cv.imread('./images/Rose.jpg') # reading the image

if img_raw is None:
    print("Image not found")
else:
    img = cv.cvtColor(img_raw,cv.COLOR_BGR2RGB) # converting BGR image to RGB
    c = 255/np.log(1 + np.max(img))
    log_img = c * (np.log(1+img))

    log_img = np.array(log_img, dtype = np.uint8)

    #combine original image and log transformed image
    combine_img = np.hstack((img,log_img))


    ''' Uncomment the below line to save the combined image of original image and log transformed image in the images folder'''
    cv.imwrite("images/log_transformed_image.jpg",np.hstack((img_raw,cv.cvtColor(log_img,cv.COLOR_RGB2BGR))))

    #show original image
    plot.imshow(combine_img)
    plot.axis("off")
    plot.title("Original_image | Log_transformed_image")
    plot.show()
