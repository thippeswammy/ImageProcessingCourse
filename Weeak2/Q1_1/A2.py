import CV2
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../../InputImg/img0.png")

# CV2.imshowC("img=", img)
# cv2.destroyAllWindows()

# Calculate intensity value counts
R_intensity_counts = [0] * 256  # Initialize a list to hold counts for each intensity value
G_intensity_counts = [0] * 256  # Initialize a list to hold counts for each intensity value
B_intensity_counts = [0] * 256  # Initialize a list to hold counts for each intensity value

# Iterate over each pixel in the image
for row in img:
    for pixel in row:
        R_intensity_counts[pixel[0]] += 1
        G_intensity_counts[pixel[1]] += 1
        B_intensity_counts[pixel[2]] += 1

color = ["Red Intensity", "Green Intensity", "Blue Intensity"]
i = -1
for intensity_counts in [R_intensity_counts, G_intensity_counts, B_intensity_counts]:
    i = i + 1
    # Display intensity value counts
    # print(intensity_counts)
    # for intensity, count in enumerate(intensity_counts):
    #     print(f"Intensity {intensity}: {count} pixels")

    # Optionally, you can plot a histogram to visualize the intensity distribution

    plt.plot(range(256), intensity_counts, color='black')
    plt.title(f'Intensity Distribution of {color[i]}')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.show()
