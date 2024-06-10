import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
image = cv2.imread('../../InputImg/img2.png', cv2.IMREAD_COLOR)

# Convert the image from BGR to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Adjusting Brightness and Contrast
# Increase contrast and brightness
alpha = 1.5  # Contrast control
beta = 50  # Brightness control
adjusted = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)

# 2. Histogram Equalization
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
equalized = cv2.equalizeHist(image_gray)

# 3. Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(image_rgb, -1, kernel)

# 4. Noise Reduction using Gaussian Blur
blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)

# Displaying the images
titles = ['Original Image', 'Brightness & Contrast', 'Histogram Equalization', 'Sharpened Image', 'Noise Reduction']
images = [image_rgb, adjusted, equalized, sharpened, blurred]

plt.figure(figsize=(15, 7))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    if len(images[i].shape) == 2:  # Grayscale image
        plt.imshow(images[i], cmap='gray')
    else:  # Color image
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
