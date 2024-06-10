import cv2
import numpy as np


def add_noise(image, noise_level):
    """Add random noise to the image."""
    noise = np.random.normal(scale=noise_level, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def main():
    # Generate a test image
    image_size = (256, 256)  # Size of the image
    test_image = np.zeros(image_size, dtype=np.uint8)

    # Add some features to the image
    cv2.circle(test_image, (100, 100), 50, 255, -1)
    cv2.rectangle(test_image, (150, 150), (200, 200), 255, -1)

    # Display the original image
    cv2.imshow("Original Image", test_image)
    cv2.waitKey(0)

    # Parameters
    max_N = 10  # Maximum number of repetitions
    noise_level = 30  # Standard deviation of the random noise

    # Repeat adding noise and summing the images
    summed_image = np.zeros_like(test_image, dtype=np.float64)
    for N in range(1, max_N + 1):
        noisy_image = add_noise(test_image, noise_level)
        summed_image += noisy_image
        cv2.imshow(f"Noisy Image (N={N})", noisy_image)
        cv2.waitKey(1000)  # Show each noisy image for 1 second

    # Normalize the summed image to display it properly
    summed_image /= max_N
    summed_image = np.clip(summed_image, 0, 255).astype(np.uint8)

    # Display the final summed image
    cv2.imshow("Summed Image", summed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
