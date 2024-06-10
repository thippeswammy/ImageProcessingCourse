import cv2
import numpy as np

# Load the image
image_rgb = cv2.imread('../../InputImg/img0.png')

# Check if the image was successfully loaded
if image_rgb is None:
    print("Error: Unable to load the image.")
    exit()

# Convert the image to 8-bit unsigned integers
image_rgb = cv2.convertScaleAbs(image_rgb)

# Convert the RGB image to YCrCb color space
image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

# Define the quantization matrices for Y, Cr, and Cb channels
Q_y = np.array([
    [16, 11, 10],
    [24, 40, 51],
    [61, 12, 14]
], dtype=np.float32)
Q_cbcr = np.array([
    [17, 18, 24],
    [47, 99, 99],
    [99, 99, 99]
], dtype=np.float32)

# Perform quantization on Y, Cr, and Cb channels
quantized_y = np.round(image_ycbcr[:, :, 0] / Q_y)
quantized_cb = np.round(image_ycbcr[:, :, 1:] / Q_cbcr)

# Invert the quantization
dequantized_y = quantized_y * Q_y
dequantized_cbcr = quantized_cb * Q_cbcr

# Combine the dequantized Y channel and dequantized CrCb channels
dequantized_ycbcr = np.zeros_like(image_ycbcr, dtype=np.float32)
dequantized_ycbcr[:, :, 0] = dequantized_y
dequantized_ycbcr[:, :, 1:] = dequantized_cbcr

# Convert the dequantized YCrCb image back to RGB color space
image_rgb_compressed = cv2.cvtColor(dequantized_ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# Display the original and compressed images
cv2.imshow('Original Image', image_rgb)
cv2.imshow('Compressed Image', image_rgb_compressed)
cv2.waitKey(0)
cv2.destroyAllWindows()
