import cv2 as cv 
import numpy as np

img = cv.imread('Pictures/resized_image.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gray', gray)


cv.imshow('Tiger', img)

# laplaction
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))


# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobley = cv.Sobel(gray, cv.CV_64F, 0, 1)

cv.imshow('Soblex', sobelx)
cv.imshow('Sobley', sobley)

# cv.imshow('Laplacian', lap)
cv.waitKey(0)

