import numpy as np
import cv2
from matplotlib import pyplot as plt


def histogram_equalization(image):
    # Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Use the CDF to map the old pixel values to new ones
    cdf_m = np.ma.masked_equal(cdf, 0)  # Mask zeros
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Normalize to [0, 255]
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # Fill masked values with zeros and convert to uint8

    image_equalized = cdf[image]

    return image_equalized


def plot(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_equalized = histogram_equalization(image)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(image_equalized, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.tight_layout()

    # Histogram of Original Image
    plt.subplot(2, 2, 3)
    plt.hist(image.flatten(), 256, [0, 256], color='black')
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Histogram of Equalized Image
    plt.subplot(2, 2, 4)
    plt.hist(image_equalized.flatten(), 256, [0, 256], color='black')
    plt.title('Histogram of Equalized Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


for i in range(1, 3):
    image_path = f'../../InputImg/img{i}.png'
    plot(image_path)
