import cv2
import numpy as np
import CV2
from scipy.fftpack import dct
from Weeak2.Q1.A1 import divide_image_into_blocks
import matplotlib.pyplot as plt


# Images = divide_image_into_blocks(img, blockSize=(64, 256), returnOnly=True)
def QuantizeImage(img, coefficient=16):
    QuantizedImg = np.zeros_like(img)
    ErrorImg = np.zeros_like(img)
    ErrorFreq = [0 for i in range(0, coefficient)]
    for i in range(img.shape[0]):
        for n in range(img.shape[1]):
            QuantizedImg[i][n] = (img[i][n] // coefficient).astype('uint8') * coefficient
            ErrorImg[i][n] = (img[i][n] - QuantizedImg[i][n]).astype('uint8')
            ErrorFreq[ErrorImg[i][n]] += 1

    CV2.imshow("InputImg", img)
    CV2.imshow("QuantizedImg", QuantizedImg)
    cv2.imwrite("QuantizedImg.png", QuantizedImg)
    CV2.imshow("ErrorImg", ErrorImg)
    cv2.imwrite("ErrorImg.png", ErrorImg)

    plt.bar(range(0, coefficient), ErrorFreq)
    plt.xlabel("Error")
    plt.ylabel("Freq")
    plt.title("ErrorImg-bar")
    plt.savefig("ErrorImg-bar.png")
    plt.show()

    plt.plot(range(0, coefficient), ErrorFreq)
    plt.xlabel("Error")
    plt.ylabel("Freq")
    plt.title("ErrorImg-plot")
    plt.savefig("ErrorImg-plot.png")
    plt.show()

    plt.scatter(range(0, coefficient), ErrorFreq, color='blue')
    plt.xlabel("Error")
    plt.ylabel("Freq")
    plt.title("ErrorImg-scatter")
    plt.savefig("ErrorImg-scatter.png")
    plt.show()

    ''' 
    # plt.hist(ErrorFreq, bins=range(coefficient + 2), edgecolor='black', align='left')
    # plt.xlabel("Error")
    # plt.ylabel("Freq")
    # plt.title("ErrorImg-hist")
    # plt.savefig("ErrorImg-hist.png")
    # plt.show()
    '''


inputImg = cv2.imread("../../InputImg/img0.png", cv2.IMREAD_GRAYSCALE)
QuantizeImage(inputImg, )
