import cv2
import argparse

# Load the image
image = cv2.imread("Pictures/spot3.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding (you can adjust the threshold value)
_, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over contours and identify bright spots
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 100:  # Adjust this threshold as needed
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, f"Spot {i+1}", (contour[0][0][0], contour[0][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow("Detected Spots", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


