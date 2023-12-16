import cv2
import numpy as np

# Read the image as a numpy array
image = cv2.imread('1.jpg')

# Apply the Canny edge detector
edges = cv2.Canny(image, 100, 200)

# Find the contours of the edges
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a new blank image to draw the vectorized image
vector_image = np.zeros_like(image)

# Draw contours on the blank image to form vectorized shapes
cv2.drawContours(vector_image, contours, -1, (255, 255, 255), 1)

# Save the vectorized image
cv2.imwrite('1.jpg', vector_image)