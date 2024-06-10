import cv2
import numpy as np

img = cv2.imread("../../InputImg/img6.png")

kernal = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
