import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def dct_quantize(channel, q_factor):
    h, w = channel.shape
    dct_quant = np.zeros((h, w), np.float32)
    block_size = 8
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i + block_size, j:j + block_size]
            dct_block = cv2.dct(block.astype(np.float32) - 128)
            dct_quant[i:i + block_size, j:j + block_size] = np.round(dct_block / q_factor)
    return dct_quant


def idct_dequantize(dct_quant, q_factor):
    h, w = dct_quant.shape
    dequant_channel = np.zeros((h, w), np.float32)
    block_size = 8
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_quant[i:i + block_size, j:j + block_size]
            dequant_block = block * q_factor
            dequant_channel[i:i + block_size, j:j + block_size] = cv2.idct(dequant_block) + 128
    return dequant_channel


def compress_image(image, q_factor_y, q_factor_cbcr):
    # Convert RGB to YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    Y, Cb, Cr = cv2.split(ycbcr_image)
    cv2.imshow("Y", Y)
    cv2.waitKey(0)
    cv2.imshow("Cb", Cb)
    cv2.waitKey(0)
    cv2.imshow("Cr", Cr)
    cv2.waitKey(0)

    # Compress the Y channel
    Y_dct_quant = dct_quantize(Y, q_factor_y)

    # Compress the Cb and Cr channels with higher compression
    Cb_dct_quant = dct_quantize(Cb, q_factor_cbcr)
    Cr_dct_quant = dct_quantize(Cr, q_factor_cbcr)

    cv2.imshow("Y_dct_quant", Y_dct_quant)
    cv2.waitKey(0)
    cv2.imshow("Cb_dct_quant", Cb_dct_quant)
    cv2.waitKey(0)
    cv2.imshow("Cr_dct_quant", Cr_dct_quant)
    cv2.waitKey(0)

    # Decompress the channels
    Y_dequant = idct_dequantize(Y_dct_quant, q_factor_y)
    Cb_dequant = idct_dequantize(Cb_dct_quant, q_factor_cbcr)
    Cr_dequant = idct_dequantize(Cr_dct_quant, q_factor_cbcr)

    # Clip values to valid range [0, 255]
    Y_dequant = np.clip(Y_dequant, 0, 255).astype(np.uint8)
    Cb_dequant = np.clip(Cb_dequant, 0, 255).astype(np.uint8)
    Cr_dequant = np.clip(Cr_dequant, 0, 255).astype(np.uint8)

    # cv2.imshow("Y_dequant", Y_dequant)
    # cv2.waitKey(0)
    # cv2.imshow("Cb_dequant", Cb_dequant)
    # cv2.waitKey(0)
    # cv2.imshow("Cr_dequant", Cr_dequant)
    # cv2.waitKey(0)

    # Merge channels and convert back to RGB
    ycbcr_reconstructed = cv2.merge([Y_dequant, Cb_dequant, Cr_dequant])
    rgb_reconstructed = cv2.cvtColor(ycbcr_reconstructed, cv2.COLOR_YCrCb2RGB)
    cv2.imshow("ycbcr_reconstructed", ycbcr_reconstructed)
    cv2.waitKey(0)
    cv2.imshow("rgb_reconstructed", rgb_reconstructed)
    cv2.waitKey(0)
    return rgb_reconstructed


# Load the image
image_path = '../../InputImg/img0.png'
image = Image.open(image_path)
image = np.array(image)

# Set the quantization factors
q_factor_y = 10
q_factor_cbcr = 50

# Compress and reconstruct the image
reconstructed_image = compress_image(image, q_factor_y, q_factor_cbcr)

# Display the original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title('Reconstructed Image')

plt.show()
