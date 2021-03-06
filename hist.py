from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('Average_uninfected.png')

chans = cv2.split(img)
colors = ('r', 'g', 'b')

plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("Pixels Intensities")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

plt.savefig('uninfected.png')
plt.show()

