import cv2
import numpy as np

# Load the image
image = cv2.imread("Pictures/spot8.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Find corners
corners = cv2.goodFeaturesToTrack(blurred, 25, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# Thresholding (you can adjust the threshold value)
_, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over contours and identify bright spots
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 100:  # Adjust this threshold as needed
        # Draw contours
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        
        # Calculate the centroid and mark it
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
            cv2.putText(image, f"Center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the result
cv2.imshow("Detected Spots and Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()