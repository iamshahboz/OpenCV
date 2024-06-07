import cv2 as cv 
import numpy as np 


'''Resize the image and save'''
# img = cv.imread('Pictures/tiger.jpg')

# width=500
# height = 400

# resized_image = cv.resize(img, (width, height))

# cv.imwrite('Pictures/resized_image.jpg', resized_image)

'''Create blank image'''
blank = np.zeros((500, 500, 3), dtype='uint8')
#cv.imshow('Blank', blank)

'''1. Paint the image certain colour'''
# blank[:] = 0, 255, 0
#cv.imshow('Green',blank)

# you can give a color to a specific area
# blank[200:300, 300:400] = 0, 0, 255
# cv.imshow('Partly colored', blank)

'''Draw a rectangle'''
# cv.rectangle(blank, (0,0), (250, 250), (0,255,0), thickness=cv.FILLED)
# cv.imshow('Rectangle', blank)

'''Draw a circle'''
#cv.circle(blank, (250, 250), 40, (0,0,255), thickness=3)
#cv.imshow('Circle',blank)

'''Draw a line'''
#cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
# cv.imshow('Line', blank)

'''Writing text  on image'''
cv.putText(blank, 'Hello', (225, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
cv.imshow('Text', blank)



cv.waitKey(0)


