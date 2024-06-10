import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.fftpack import dct, idct

# Load an example image and convert it to grayscale
image = color.rgb2gray(data.astronaut())


# Function to apply DCT to an image
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


# Function to apply inverse DCT to an image
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# Function to apply DCT to the entire image
def apply_dct(image, block_size):
    h, w = image.shape
    dct_image = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dct_image[i:i + block_size, j:j + block_size] = dct2(image[i:i + block_size, j:j + block_size])
    return dct_image


# Function to quantize the DCT coefficients
def quantize(dct_image, Q, block_size):
    h, w = dct_image.shape
    quantized_image = np.zeros_like(dct_image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            quantized_image[i:i + block_size, j:j + block_size] = dct_image[i:i + block_size, j:j + block_size] // Q
    return quantized_image


# Function to dequantize the quantized DCT coefficients
def dequantize(quantized_image, Q, block_size, dct_image):
    h, w = quantized_image.shape
    dequantized_image = np.zeros_like(quantized_image)
    error_image = np.zeros_like(dct_image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dequantized_image[i:i + block_size, j:j + block_size] = quantized_image[i:i + block_size,
                                                                    j:j + block_size] * Q
            error_image[i:i + block_size, j:j + block_size] = dct_image[i:i + block_size,
                                                              j:j + block_size] - dequantized_image[i:i + block_size,
                                                                                  j:j + block_size]
    return dequantized_image, error_image


# Function to apply inverse DCT to the entire image
def apply_idct(dct_image, block_size):
    h, w = dct_image.shape
    idct_image = np.zeros_like(dct_image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            idct_image[i:i + block_size, j:j + block_size] = idct2(dct_image[i:i + block_size, j:j + block_size])
    return idct_image


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

# Quantize the DCT coefficients
quantized_image = quantize(dct_image, Q, block_size=8)

# Dequantize the coefficients
deQuantized_image, ErrorImg = dequantize(quantized_image, Q, block_size=8, dct_image=dct_image)

# Perform inverse DCT to reconstruct the image
reconstructed_image_dct = apply_idct(deQuantized_image, block_size=8)

# Display all images in one plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("DCT Image")
plt.imshow(dct_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Quantized DCT Image")
plt.imshow(quantized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Dequantized Image")
plt.imshow(deQuantized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Error Image")
plt.imshow(ErrorImg, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image_dct, cmap='gray')
plt.axis('off')

plt.show()

# Check if the original and reconstructed images are the same
print("Reconstruction successful:", np.allclose(image, reconstructed_image_dct, atol=1e-1))
