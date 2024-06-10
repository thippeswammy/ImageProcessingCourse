import cv2
import numpy as np

import CV2
from scipy.fftpack import dct
from Weeak2.Q1.A1 import divide_image_into_blocks


def discreteCosineTransform(images):
    imagesDCT = []
    for img_ in images[:]:
        img_dct = dct(dct(img_.T, norm='ortho').T, norm='ortho')
        imagesDCT.append(img_dct)
    return imagesDCT


img = cv2.imread("../../InputImg/img0.png", cv2.IMREAD_GRAYSCALE)
DCT_img = discreteCosineTransform(divide_image_into_blocks(img, blockSize=(256, 256), returnOnly=True))
block = 256
for i, img in enumerate(DCT_img):
    if i < 5:
        CV2.imshowC(f"DCT_img{i}", img)
    cv2.imwrite(f"DCT_img{i}.png", img)
