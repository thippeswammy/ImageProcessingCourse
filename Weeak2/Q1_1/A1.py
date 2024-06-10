import CV2
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../../InputImg/img0.png", cv2.IMREAD_GRAYSCALE)

CV2.imshowC("img=", img)
cv2.destroyAllWindows()

# Calculate intensity value counts
intensity_counts = [0] * 256  # Initialize a list to hold counts for each intensity value

# Iterate over each pixel in the image
for row in img:
    for pixel in row:
        intensity_counts[pixel] += 1

# Display intensity value counts
for intensity, count in enumerate(intensity_counts):
    print(f"Intensity color {intensity}: {count} pixels")

# Optionally, you can plot a histogram to visualize the intensity distribution

plt.plot(range(256), intensity_counts, color='black')
plt.title('Intensity Color Distribution')
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.show()
