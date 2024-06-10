import numpy as np
import cv2
from matplotlib import pyplot as plt


def compute_histogram(image):
    # Calculate histogram for each channel
    hist = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(image.shape[2])]
    return hist


def compute_prediction_errors(image):
    # Initialize an array to store prediction errors
    prediction_errors = np.zeros_like(image, dtype=np.int16)

    # Iterate over the image pixels
    for y in range(image.shape[0]):
        for x in range(1, image.shape[1]):  # Start from 1 to avoid out-of-bounds
            prediction_errors[y, x] = image[y, x] - image[y, x - 1]
    return prediction_errors


def plot_histograms(image, prediction_errors):
    # Compute histograms
    hist_image = compute_histogram(image)

    # Shift to [0, 255] range and convert to uint8
    prediction_errors_shifted = np.clip(prediction_errors + 128, 0, 255).astype(np.uint8)
    hist_errors = compute_histogram(prediction_errors_shifted)

    # Plot histograms
    plt.figure(figsize=(12, 6))

    # Original Image Histogram
    plt.subplot(2, 1, 1)
    for i, col in enumerate(['b', 'g', 'r']):
        plt.plot(hist_image[i], color=col)
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Prediction Errors Histogram
    plt.subplot(2, 1, 2)
    for i, col in enumerate(['b', 'g', 'r']):
        plt.plot(hist_errors[i], color=col)
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Prediction Error Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Load the image
image_path = '../../InputImg/img0.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with Matplotlib

# Compute prediction errors
prediction_errors = compute_prediction_errors(image)

# Plot histograms
plot_histograms(image, prediction_errors)
