import numpy as np
import cv2

import cv2

import CV2

# Load the image
image = cv2.imread('../../InputImg/G1.jpg', cv2.IMREAD_COLOR)


# image = cv2.resize(image,None,fx=0.4,fy=0.4)
# CV2.imshow("OriginalImage",image)
# cv2.destroyAllWindows()


# Function to reduce spatial resolution
def reduce_resolution(image, block_size):
    height, width, channels = image.shape
    new_image = np.zeros((height // block_size, width // block_size, channels), dtype=np.uint8)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Use floor division to ensure indices are within bounds
            block = image[i:i + block_size, j:j + block_size]
            average_color = np.mean(block, axis=(0, 1))
            new_image[i // block_size - 1, j // block_size - 1] = average_color.astype(np.uint8)
    return new_image


# Reduce resolution for different block sizes
image_3x3 = reduce_resolution(image, 3)
image_5x5 = reduce_resolution(image, 5)
image_7x7 = reduce_resolution(image, 7)

CV2.imshowC("image_3x3", image_3x3)

CV2.imshowC("image_5x5", image_5x5)

CV2.imshowC("image_7x7", image_7x7)

# Save the images
cv2.imwrite('image_3x3.jpg', image_3x3)
cv2.imwrite('image_5x5.jpg', image_5x5)
cv2.imwrite('image_7x7.jpg', image_7x7)
