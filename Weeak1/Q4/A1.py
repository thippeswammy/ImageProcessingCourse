import cv2
import numpy as np

import CV2


def reduce_resolution(image, block_sizes=(3, 5, 7)):
    """
    Reduces the spatial resolution of an image by averaging pixel values within blocks.
    Args:
        image: The input image as a NumPy array.
        block_sizes: A tuple of block sizes (e.g., (3, 5, 7)) to average over.
    Returns:
        A list of down sampled images, one for each block size.
    """

    downSampledImagess = []
    for block_size in block_sizes:
        height, width, channels = image.shape

        # Calculate number of blocks
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size

        print(num_blocks_h * block_size, " ", height, " ", num_blocks_w * block_size, " ", width)
        # Reshape into blocks and calculate averages
        downSampledBlock = np.mean(image[:num_blocks_h * block_size, :num_blocks_w * block_size].reshape(
            num_blocks_h, block_size, num_blocks_w, block_size, channels), axis=(1, 3))

        downSampledImagess.append(downSampledBlock.astype(image.dtype))

    return downSampledImagess


# Example usage
img = cv2.imread("../../InputImg/G1.jpg")
downSampledImages = reduce_resolution(img)

block_size = [3, 5, 7]
# Display or save the downSampled images
for i, img in enumerate(downSampledImages):
    CV2.imshowC(f"DownSampled (block size {block_size[i]})", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
