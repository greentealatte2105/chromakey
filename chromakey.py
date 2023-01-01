import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Flatten the image into a 2D array of pixels
pixels = hsv.reshape((-1, 3))

# Use k-means clustering to cluster the pixels into 2 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(pixels)

# Identify the cluster that represents the background (assume it is the cluster with the most pixels)
labels = kmeans.labels_
background_cluster = np.argmax(np.bincount(labels))

# Create a mask that only keeps the pixels from the background cluster
mask = np.where(labels == background_cluster, 0, 255).astype(np.uint8)
mask = mask.reshape(image.shape[:2])

# Create a transparent image with an alpha channel
transparent_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
transparent_image[:,:,:3] = image
transparent_image[:,:,3] = mask



# Set the pixels outside the objects to fully transparent
transparent_image[mask == 0] = (255, 255, 255, 0)
# transparent_image = cv2.cvtColor(transparent_image, cv2.COLOR_BGR2RGBA)

cv2.imwrite('final_image.png', transparent_image)
