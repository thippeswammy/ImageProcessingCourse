import cv2

import cv2

import CV2

# Read the input image
# It is color is from (0-255) black or white
# 0(000) -> black; 1(255) -> white , 8-Bit
input_image = cv2.imread('../../InputImg/G1.jpg', cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, None, fx=0.4, fy=0.4)

# Display the original image
CV2.imshow('Original Image', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
i = 0
while i < 20:
    i = i + 1

    # Calculate the scaling factor to reduce intensity levels
    scaling_factor = 2 ** 8 / 2 ** (i - 1)
    print(i, ";  ", scaling_factor, ";  ", ((2 ** i) - 1), ";  ", 255 / scaling_factor)
    # Perform intensity level reduction
    output_image = (input_image / scaling_factor).astype('uint8') * scaling_factor

    # Display the processed image
    CV2.imshowC(f'Processed Image{i}', output_image)

    # Save the processed image
    cv2.imwrite(f'G1_Output{i}.jpg', output_image)

cv2.destroyAllWindows()
