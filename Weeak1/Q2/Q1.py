import numpy as np

import cv2

import CV2

# Load the image
image = cv2.imread('../../InputImg/G1.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, None, fx=0.4, fy=0.4)
CV2.imshowC(f'OriginalImage', image)

# Define neighborhood sizes
neighborhood_sizes = [3, 5, 10, 20, 30, 40]

# Perform spatial averaging for each neighborhood size
for size in neighborhood_sizes:
    # Define the kernel for averaging
    kernel = np.ones((size, size), np.float32) / (size * size)
    # print(kernel)

    # Apply the filter
    averaged_image = cv2.filter2D(image, -1, kernel)

    # Display the result
    CV2.imshowC(f'Averaged Image ({size}x{size} Neighborhood)', averaged_image)

cv2.destroyAllWindows()
