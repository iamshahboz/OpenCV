import cv2 as cv 
import numpy as np 

img = cv.imread('Pictures/resized_image.jpg')

'''
Translation
It is shifting an image along the x and y axis 

'''

def translate(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimentions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimentions)

# -x ---> Left
# -y ---> Up 
# x ----> Right
# y ----> Down 

# translated = translate(img, -100, 100)
# cv.imshow('Translated', translated)

'''
Rotation
'''
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (width//2, height//2)
        
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimentions = (width, height)
    
    return cv.warpAffine(img, rotMat, dimentions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

cv.waitKey(0)
