import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_salt_and_pepper_noise(image):
    noisy_image = np.copy(image)

    salt_prob = 0.02  # Probability of salt noise
    pepper_prob = 0.02  # Probability of pepper noise

    num_salt = np.ceil(salt_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 1

    num_pepper = np.ceil(pepper_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image


def show(image_rgb, noisy_image, denoised_image):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image)
    plt.title('Image with Salt and Pepper Noise')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image)
    plt.title('Denoised Image with Median Filter')
    plt.axis('off')

    plt.show()


image = cv2.imread('../../InputImg/img2.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

noisy_image = add_salt_and_pepper_noise(image_rgb)
# noisy_image = image_rgb

denoised_image = cv2.medianBlur(noisy_image, 3)

show(image_rgb, noisy_image, denoised_image)
