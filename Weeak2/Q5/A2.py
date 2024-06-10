import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load an example image and convert it to grayscale
image = color.rgb2gray(data.astronaut())

# Define the quantization matrix
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Repeat the quantization matrix to match the image size
quantization_matrix = np.kron(np.ones((64, 64)), Q)
# quantization_matrix = np.repeat(np.repeat(Q, image.shape[0] // 8, axis=0), image.shape[1] // 8, axis=1)

# Quantize the original image
quantized_image = np.round(image*255 / quantization_matrix)

# Dequantize the quantized image
dequantized_image = (quantized_image * quantization_matrix).astype("uint8")

# Calculate the error between the dequantized image and the original image
error_image = image - dequantized_image

# Reconstruct the image using the dequantized image
reconstructed_image = dequantized_image

# Display all images in one plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Quantized Image")
plt.imshow(quantized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Dequantized Image")
plt.imshow(dequantized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Error Image")
plt.imshow(error_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.show()
