import cv2
import CV2


def divide_image_into_blocks(image, blockSize=(128, 128), returnOnly=True):
    w, h = image.shape[:2]
    images = []
    for i in range(0, h, blockSize[0]):
        for n in range(0, w, blockSize[1]):
            subImg = image[i:i + blockSize[0], n:n + blockSize[1]]
            images.append(subImg)
    if returnOnly:
        return images
    for count, subImg in enumerate(images):
        cv2.imwrite(f"SubImg{count}.png", subImg)
        # CV2.imshowC(f"sub Image {count}", img)


if __name__ == "__main__":
    img = cv2.imread("../../InputImg/img0.png", cv2.IMREAD_GRAYSCALE)
    CV2.imshowC("input img", img)
