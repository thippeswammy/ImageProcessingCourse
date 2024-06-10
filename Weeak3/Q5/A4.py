import cv2


def color_edge_detector(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges


for i in range(0, 8):
    image = cv2.imread(f"../../InputImg/img{i}.png")

    edge_mask = color_edge_detector(image)
    cv2.imshow("Original Image", image)
    cv2.imshow("Color Edges", edge_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
