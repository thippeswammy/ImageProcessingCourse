import numpy as np
import matplotlib.pyplot as plt


def hist(image):
    hist = np.array([0 for i in range(0, 256)])
    for i in range(image.shape[0]):
        for n in range(image.shape[1]):
            hist[image[i][n]] += 1
    return hist


def HistPlot(name, hist):
    plt.figure(figsize=(12, 6))
    plt.plot(hist, color="black")
    plt.title(f'Histogram {name}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
