import cv2 as cv  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plot #Import matplotlib.pyplot for ploting the image

def mean_filter(image, kernel_size):
    """
    Applies a mean (average) filter to the input image using convolution.
    
    Parameters:
        image (numpy.ndarray): Grayscale image to be filtered.
        kernel_size (int): Size of the square filter kernel.

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Create a kernel (filter) with equal weights (each pixel gets equal importance)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # Apply convolution using OpenCV's filter2D function
    # -1 in the second argument means the output has the same depth as the input image
    filtered_image = cv.filter2D(image, -1, kernel)
    
    return filtered_image  # Return the processed image

def resize_image(image, max_size=500):
    """
    Resizes the image while maintaining the original aspect ratio.

    Parameters:
        image (numpy.ndarray): Image to be resized.
        max_size (int): Maximum allowed width or height for the resized image.

    Returns:
        numpy.ndarray: Resized image with preserved aspect ratio.
    """
    # Get original dimensions of the image
    h, w = image.shape[:2]  
    
    # Compute the scale factor that ensures the largest dimension fits within max_size
    scale = min(max_size / w, max_size / h)  
    
    # Calculate the new dimensions using the computed scale factor
    new_w, new_h = int(w * scale), int(h * scale)  
    
    # Resize the image using INTER_AREA interpolation (good for downscaling)
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

# image = cv.imread('./images/Image17.webp')
# borderd_image = cv.copyMakeBorder(resize_image(image, max_size=500),10,10,10,10,cv.BORDER_CONSTANT,value=(50,18,255))


# Load the input image in grayscale mode (0-255 intensity values)
image = cv.imread('./images/Girl.png', cv.IMREAD_GRAYSCALE) 


# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")  # Print error message if loading fails
    exit()  # Exit the program

# Define the kernel size for the mean filter (commonly 3×3 for smoothing)
kernel_size = 3 

# Apply the mean filter to the image
filtered_image = mean_filter(image, kernel_size)  

# Resize both images to fit within 500×500 pixels while maintaining aspect ratio
resized_original = resize_image(image, max_size=500)  
resized_filtered = resize_image(filtered_image, max_size=500)  

# Combine both images side by side using np.hstack() (horizontal stacking)
combined_image = np.hstack((resized_original, resized_filtered))

'''Uncomment the below line to save the meanfiltered image in the images folder'''
# cv.imwrite("images/meanfiltered_image.jpg", combined_image)

# Display the original and filtered images in a single window
plot.imshow(combined_image,cmap="gray") # cmap = "gray" since the image is in BGR format
plot.axis("off") #hide axis
plot.title("Original_Image | Meanfiltered_Image")
plot.show()
