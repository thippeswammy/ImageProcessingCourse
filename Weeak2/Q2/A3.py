import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.fftpack import dct, idct


# Function to apply DCT to an image
def apply_dct(image, block_size):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')


# Function to apply inverse DCT to an image
def apply_idct(dct_image, block_size):
    return idct(idct(dct_image, axis=0, norm='ortho'), axis=1, norm='ortho')


# Quantize the DCT coefficients
def quantize(dct_image, Q):
    # Repeat the quantization matrix to match the shape of the DCT image
    quantization_matrix = np.repeat(np.repeat(Q, 64, axis=0), 64, axis=1)
    # Perform quantization
    return np.round(dct_image / quantization_matrix)


# Dequantize the coefficients
def dequantize(quantized_image, Q):
    # Repeat the quantization matrix to match the shape of the quantized image
    quantization_matrix = np.repeat(np.repeat(Q, 64, axis=0), 64, axis=1)
    # Perform dequantization
    return quantized_image * quantization_matrix


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

# Perform DCT
dct_image = apply_dct(image, block_size=8)

# Perform inverse DCT to reconstruct the image without quantization
reconstructed_image_dct_without_quantization = apply_idct(dct_image, block_size=8)

# Quantize the DCT coefficients
quantized_image = quantize(dct_image, Q)

# Dequantize the coefficients
deQuantized_image = dequantize(quantized_image, Q)

# Perform inverse DCT to reconstruct the image with quantization
reconstructed_image_dct_with_quantization = apply_idct(deQuantized_image, block_size=8)

# Calculate the absolute difference between the original and reconstructed images with quantization
difference_quantized = np.abs(image - reconstructed_image_dct_with_quantization)

# Display all images in one plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Reconstructed Image without Quantization")
plt.imshow(reconstructed_image_dct_without_quantization, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Reconstructed Image with Quantization")
plt.imshow(reconstructed_image_dct_with_quantization, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Absolute Difference")
plt.imshow(difference_quantized, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Quantized Image")
plt.imshow(quantized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Dequantized Image")
plt.imshow(deQuantized_image, cmap='gray')
plt.axis('off')

plt.show()

# Check if the original and reconstructed images with quantization are approximately equal
reconstruction_successful = np.allclose(image, reconstructed_image_dct_with_quantization, atol=1)

print("Reconstruction with quantization successful:", reconstruction_successful)
