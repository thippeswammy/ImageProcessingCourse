import cv2

import CV2

# Read the input image
# It is color is from (0-255) black or white
# 0(000) -> black; 1(255) -> white , 8-Bit
input_image = cv2.imread('../../InputImg/G1.jpg', cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, None, fx=0.4, fy=0.4)

# Display the original image
CV2.imshowC('Original Image', input_image)
cv2.destroyAllWindows()

# Number of intensity levels to reduce to
InputImageNumbersOfBitsPerPixel = 8
InputImageColorLevelsPerPixel = 256

desired_levels = 1
while InputImageNumbersOfBitsPerPixel - desired_levels > 0:
    # Calculate the scaling factor to reduce intensity levels
    scaling_factor = 2 ** desired_levels
    # print(i, ";  ", scaling_factor + 1, ";  ")
    # Perform intensity level reduction
    output_image = (input_image / scaling_factor).astype('uint8') * scaling_factor
    # Display the processed image
    CV2.imshowC(
        f'ProcessedImage={InputImageColorLevelsPerPixel}/{scaling_factor}  ColorLevelsPerPixel={(InputImageColorLevelsPerPixel) // scaling_factor}'
        f'  NumbersOfBitsPerPixel={(InputImageNumbersOfBitsPerPixel - desired_levels)}',
        output_image)

    # Save the processed image
    cv2.imwrite(
        f'ProcessedImage={InputImageColorLevelsPerPixel}By{scaling_factor}  ColorLevelsPerPixel={(InputImageColorLevelsPerPixel) // scaling_factor}'
        f'  NumbersOfBitsPerPixel={(InputImageNumbersOfBitsPerPixel - desired_levels)}.jpg',
        output_image)
    desired_levels += 1

cv2.destroyAllWindows()
