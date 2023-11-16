import cv2
import numpy as np

# Load the image
image = cv2.imread('bear.png')

# Calculate the average values of the R, G, and B channels
average_r = np.mean(image[:, :, 2])
average_g = np.mean(image[:, :, 1])
average_b = np.mean(image[:, :, 0])

# Print the average values


average_intensity = (average_r + average_g + average_b) / 3
print(average_intensity)
if average_intensity > 55:
    print("Image is bright")
else:
    print("Image is darker")