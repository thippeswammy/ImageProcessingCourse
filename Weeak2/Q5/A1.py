import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.fft import fftn, ifftn


# Function to apply FFT to an image
def apply_fft(image):
    return fftn(image)


# Function to apply inverse FFT to an image
def apply_ifft(fft_image):
    return np.real(ifftn(fft_image))


# Quantize the FFT coefficients
def quantize(fft_image, Q):
    # Repeat the quantization matrix to match the shape of the FFT image
    quantization_matrix = np.repeat(np.repeat(Q, fft_image.shape[0] // 8, axis=0), fft_image.shape[1] // 8, axis=1)
    # Perform quantization
    return np.round(fft_image / quantization_matrix)


# Dequantize the coefficients
def dequantize(quantized_image, Q):
    # Repeat the quantization matrix to match the shape of the quantized image
    quantization_matrix = np.repeat(np.repeat(Q, quantized_image.shape[0] // 8, axis=0), quantized_image.shape[1] // 8,
                                    axis=1)
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

# Perform FFT
fft_image = abs(apply_fft(image))
ma = int(np.max(fft_image))
fft_image1 = (fft_image * 255) // ma
cv2.imshow("fft_image1", fft_image1)
cv2.waitKey(0)
# Quantize the FFT coefficients
quantized_image = quantize(fft_image, Q)
ma1 = int(np.max(quantized_image))
quantized_image1 = (quantized_image * 255) // ma1
cv2.imshow("quantized_image1", quantized_image1)
cv2.waitKey(0)
# Dequantize the coefficients
dequantized_image = dequantize(quantized_image, Q)
cv2.imshow("dequantized_image", dequantized_image)
cv2.waitKey(0)

# Perform inverse FFT to reconstruct the image with quantization
reconstructed_image_fft_with_quantization = apply_ifft(dequantized_image)
cv2.imshow("reconstructed_image_fft_with_quantization", reconstructed_image_fft_with_quantization)
cv2.waitKey(0)

# Display all images in one plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("FFT Magnitude")
plt.imshow(np.abs(fft_image1), cmap='gray', vmax=np.abs(fft_image1).max())
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Quantized FFT Magnitude")
plt.imshow(np.abs(quantized_image1), cmap='gray', vmax=np.abs(quantized_image1).max())
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Dequantized FFT Magnitude")
plt.imshow(np.abs(dequantized_image), cmap='gray', vmax=np.abs(dequantized_image).max())
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Reconstructed Image with Quantization (FFT)")
plt.imshow(reconstructed_image_fft_with_quantization, cmap='gray')
plt.axis('off')

plt.show()

# Check if the original and reconstructed images with quantization are approximately equal
reconstruction_successful = np.allclose(image, reconstructed_image_fft_with_quantization, atol=1)

print("Reconstruction with quantization using FFT successful:", reconstruction_successful)
