import cv2
import numpy as np
import CV2
from scipy.fftpack import dct
from Weeak2.Q1.A1 import divide_image_into_blocks
import matplotlib.pyplot as plt


def FindingNthBig(img, N=8):
    num = img.flatten().tolist()
    num.sort()
    num = list(set(num))
    if N > len(num):
        return num[-1]
    return num[N - 1]


def QuantizeEachBlockPreserving8LargestCoefficients(image, blockSize=(64, 64)):
    w, h = image.shape[:2]
    QuantizedImg = np.zeros_like(image)
    ErrorImg = np.zeros_like(image)
    val = 256
    ErrorFreq = [0 for i in range(0, val)]
    for i in range(0, h, blockSize[0]):
        for n in range(0, w, blockSize[1]):
            subImg = image[i:i + blockSize[0], n:n + blockSize[1]]
            coefficient = FindingNthBig(subImg)
            QuantImg = (subImg // coefficient).astype("uint8") * coefficient
            QuantizedImg[i:i + blockSize[0], n:n + blockSize[1]] = QuantImg
            ErrorImg[i:i + blockSize[0], n:n + blockSize[1]] = (subImg - QuantImg).astype("uint8")
            for row in ErrorImg.tolist():
                for col in row:
                    ErrorFreq[col] += 1

    CV2.imshow("InputImg", image)
    CV2.imshow("QuantizedImg", QuantizedImg)
    cv2.imwrite("QuantizedImg.png", QuantizedImg)
    CV2.imshow("ErrorImg", ErrorImg)
    cv2.imwrite("ErrorImg.png", ErrorImg)

    plt.bar(range(0, val), ErrorFreq)
    plt.xlabel("Error")
    plt.ylabel("Freq")
    plt.title("ErrorImg-bar")
    plt.savefig("ErrorImg-bar.png")
    plt.show()


InputImg = cv2.imread("../../InputImg/img0.png", cv2.IMREAD_GRAYSCALE)

QuantizeEachBlockPreserving8LargestCoefficients(InputImg)
