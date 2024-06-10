import numpy as np
import cv2
from matplotlib import pyplot as plt


def compute_histogramImage(image):
    # Calculate histogram for the channel
    hist = [cv2.calcHist([image], [0], None, [255], [0, 256])]
    return hist[0]


def compute_histogramImg(image):
    # Calculate histogram for the channel
    hist = np.array([0 for i in range(0, 256)])
    for i in range(image.shape[0]):
        for n in range(image.shape[1]):
            hist[image[i][n]] += 1
    return hist


def compute_histogramError(image):
    # Calculate histogram for the channel
    hist = np.array([0 for i in range(0, 600)])
    for i in range(image.shape[0]):
        for n in range(image.shape[1]):
            hist[image[i][n] + 300] += 1
    return hist


def compute_prediction_errors(image):
    # Initialize an array to store prediction errors
    prediction_errors = np.zeros_like(image, dtype=np.int16)

    # Convert image to int16 to prevent overflow
    image = image.astype(np.int16)

    # Iterate over the image pixels
    for y in range(1, image.shape[0]):
        for x in range(1, image.shape[1]):  # Start from 1 to avoid out-of-bounds
            prediction_errors[y, x] = image[y, x] - (image[y, x - 1] + image[y - 1, x] + image[y - 1, x - 1]) // 3

    # Display prediction errors for debugging
    cv2.imshow("prediction_errors", abs(prediction_errors.astype(np.uint8)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return prediction_errors


def plot_histograms(image, prediction_errors):
    # Compute histograms
    hist_image = compute_histogramImg(image)

    # Shift to [0, 255] range and convert to uint8
    prediction_errors_shifted = prediction_errors.astype(np.uint8)
    hist_errors = compute_histogramError(prediction_errors_shifted)

    # Plot histograms
    plt.figure(figsize=(12, 6))

    # Original Image Histogram
    plt.subplot(2, 1, 1)
    plt.plot(hist_image, color="black")
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Prediction Errors Histogram
    plt.subplot(2, 1, 2)
    plt.plot(hist_errors, color="black", )
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Prediction Error Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Load the image
image_path = '../../InputImg/img0.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Compute prediction errors
prediction_errors = compute_prediction_errors(image)

# Plot histograms
plot_histograms(image, prediction_errors)
