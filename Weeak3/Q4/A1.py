import cv2
import numpy as np


def non_local_means(img, h, window_size):
    padded_img = cv2.copyMakeBorder(img, window_size // 2, window_size // 2, window_size // 2, window_size // 2,
                                    cv2.BORDER_REFLECT)
    denoised_img = np.zeros_like(img, dtype=np.float64)

    weight_sum = np.zeros_like(img, dtype=np.float64)

    for i in range(window_size // 2, img.shape[0] + window_size // 2):
        for j in range(window_size // 2, img.shape[1] + window_size // 2):
            patch = padded_img[i - window_size // 2:i + window_size // 2 + 1,
                    j - window_size // 2:j + window_size // 2 + 1]
            diff = patch - padded_img[i, j]
            weight = np.exp(-np.sum(diff ** 2) / (h ** 2))
            denoised_img[i - window_size // 2, j - window_size // 2] += patch[
                                                                            window_size // 2, window_size // 2] * weight
            weight_sum[i - window_size // 2, j - window_size // 2] += weight

    denoised_img /= (weight_sum + 1e-6)  # Add a small value to avoid division by zero
    return denoised_img.astype(np.uint8)


# Load image
image = cv2.imread("../../InputImg/img8.png", cv2.IMREAD_GRAYSCALE)

# Add noise to the image
noisy_image = cv2.add(image, np.random.normal(scale=30, size=image.shape).astype(np.uint8))

# Denoise the image using non-local means with different window sizes and noise levels
window_sizes = [3, 5, 7]
noise_levels = [10, 30, 50]

for window_size in window_sizes:
    for noise_level in noise_levels:
        denoised_image = non_local_means(noisy_image, h=noise_level, window_size=window_size)
        cv2.imshow(f"Denoised Image (Window Size: {window_size}, Noise Level: {noise_level})", denoised_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

