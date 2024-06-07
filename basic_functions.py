import cv2 as cv 
import numpy as np 

img = cv.imread('Pictures/resized_image.jpg')

'''Converting the image to grayscale '''

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

cv.waitKey(0)

