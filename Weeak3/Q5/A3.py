import cv2
import numpy as np


def color_edge_detector(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator to detect edges
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Threshold the gradient magnitude to obtain edges
    threshold = 50
    edge_mask = (gradient_magnitude > threshold).astype(np.uint8) * 255

    return edge_mask


for i in range(0, 8):
    # Read the input image
    image = cv2.imread(f"../../InputImg/img{i}.png")

    # Apply color edge detection
    edge_mask = color_edge_detector(image)

    # Display the original image and the detected edges
    cv2.imshow("Original Image", image)
    cv2.imshow("Color Edges", edge_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
