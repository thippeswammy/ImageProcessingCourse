from PIL import Image


def divide_image_into_blocks(image_path, block_size=(8, 8)):
    # Load the image
    image = Image.open(image_path)

    # Get image dimensions
    width, height = image.size

    # Prepare a list to store the Images
    Images = []

    # Loop over the image in steps of the block size
    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            # Define the box for the current block
            box = (x, y, x + block_size[0], y + block_size[1])
            # Extract the block
            sunImg = image.crop(box)
            # Append the block to the list
            Images.append(sunImg)

    return Images


# Example usage
image_path = 'path_to_your_image.jpg'
blocks = divide_image_into_blocks(image_path)

# To save the blocks as individual images
for i, block in enumerate(blocks):
    block.save(f'block_{i}.png')
