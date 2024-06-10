import cv2

import CV2

image = cv2.imread("../../InputImg/G1.jpg")
image = cv2.resize(image, None, fx=0.2, fy=0.2)
CV2.imshowC("OriginalImage", image)

image = cv2.imread("../../InputImg/G1.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, None, fx=0.2, fy=0.2)
CV2.imshowC("OriginalGrayImage", image)

cv2.destroyAllWindows()

# Rotate by 45 degrees
rows, cols = image.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotatedImage_45 = cv2.warpAffine(image, M, (cols, rows))

# Rotate by 90 degrees
rotatedImage_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Display the rotated images
CV2.imshow('Rotated by 45 degrees', rotatedImage_45)
CV2.imshow('Rotated by 90 degrees', rotatedImage_90)

cv2.destroyAllWindows()
