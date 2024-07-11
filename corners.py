# working version
import numpy as np
import cv2 as cv
import random

path = f'captures/image_{random.randint(1,100)}.png'
img = cv.imread('Pictures/spot8.jpg')



# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
gray = np.float32(gray)

# find light spot
_, thresholded = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)

# Find contours
contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# for i, contour in enumerate(contours):
#     area = cv.contourArea(contour)
#     if area > 100:  # Adjust this threshold as needed
#         cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
#         cv.putText(img, f"Spot {i+1}", (contour[0][0][0], contour[0][0][1]),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if area > 100:  # Adjust this threshold as needed
        cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
        
        # Calculate the centroid of the contour
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Mark the center of the light spot
            # cv.circle(img, (cX, cY), 5, (255, 0, 0), -1)
            # Print the location of the light spot
            

            
            # Print the distance
            print(f"Distance from center to Light Spot {i+1}: {distanceCm:.2f} cm")
            
            print(f"Light Spot {i+1} Location: ({cX}, {cY}) pixels")
        
        cv.putText(img, f"Spot {i+1}", (contour[0][0][0], contour[0][0][1]),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# Detect corners using cv.cornerHarris
corners = cv.cornerHarris(gray, 2, 3, 0.05)
corners = cv.dilate(corners, None)

# Mark corners on the image
img[corners > 0.01 * corners.max()] = [0, 0, 255]

# Find the center of the image
height, width = img.shape[:2]
center = (width // 2, height // 2)
print(f'The center coordinates are {center}')

# Mark the center with a red dot
cv.circle(img, center, 5, (0, 0, 255), -1)



# Display image with corners and center marked
cv.imshow('Image with Corners and Center', img)
cv.waitKey(0)

# Save the image with detected corners and center marked
cv.imwrite(path, img)
print(f"Saved image with corners and center to {path}")






