import cv2
import numpy as np

img = cv2.imread("../../InputImg/img3.png")
cv2.imshow("input image", img)
size = 8
kernel = np.ones((size, size), np.float32) / (size * size)
# Apply the filter
averaged_image = cv2.filter2D(img, -1, kernel)
averaged_imageZeros = np.zeros_like(img)
averaged_imageZeros[averaged_image > 127] = 255
cv2.imshow("averaged_image", averaged_image)
cv2.imshow("averaged_imageZeros", averaged_imageZeros)
cv2.waitKey(0)
