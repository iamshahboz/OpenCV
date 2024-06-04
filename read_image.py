# read image in open cv 
import cv2 as cv 

# the image size should be small, other it will go off your screen
img = cv.imread('Pictures/tiger.jpg')
cv.imshow('Nature', img)
cv.waitKey(0)
