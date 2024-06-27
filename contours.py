import cv2 as cv

img = cv.imread('Pictures/resized_image.jpg')

# we can convert the image to grayscale

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gray', gray)

# lets say I want to grab the edges of the image
# we can use the canny edge detector

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# lets find the contours of the image 

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} countours found in the image')





cv.imshow('Image', img)

cv.waitKey(0)

