# working version
import numpy as np
import cv2 as cv
import random

path = f'captures/image_{random.randint(1,100)}.png'
img = cv.imread('Pictures/odqa_resized.jpg')

# img = cv.imread('Pictures/new.jpg')



def order_points(points):
    """
    This function orders a list of four points into top-left, top-right, bottom-right, and bottom-left order.

    Args:
        points: A list of four tuples representing (x, y) coordinates.

    Returns:
        A list of four tuples representing the ordered points (TL, TR, BR, BL).
    """
    if len(points) == 1:
        return points

    # Sort the points based on x
    sorted_points = sorted(points, key=lambda p: p[0])

    # Get the first two points (with the lowest x-coordinate)
    leftmost = sorted_points[:2]

    # Get the last two points (with the highest x-coordinate)
    rightmost = sorted_points[2:]

    # Get the top and bottom points from each half based on y-coordinate
    top_left, bottom_left = sorted(leftmost, key=lambda p: p[1])
    top_right, bottom_right = sorted(rightmost, key=lambda p: p[1])

    # Return the ordered points
    #   return top_left, top_right, bottom_right, bottom_left
    return np.float32([top_left, top_right, bottom_right, bottom_left])

def reproject_to_square(img: np.ndarray, points):
    """
    This function reprojects an image such that the provided four points form a square.

    Args:
        image_path: Path to the image file.
        points: A list of four (x, y) coordinates representing the points to form the square.
        clockwise, starting BR, BL, TR, TL (why tho?)

    Returns:
        A NumPy array representing the reprojected image.
    """

    # Extract source points (points defining the quadrilateral in the original image)
    src_points = np.array(points, dtype=np.float32)

    # Define destination points (forming a square)
    dst_width = max(point[0] for point in points) - min(point[0] for point in points)
    dst_height = max(point[1] for point in points) - min(point[1] for point in points)
    dst_points = np.float32(
        [[dst_width, dst_height], [0, dst_height], [dst_width, 0], [0, 0]]
    )
    # print(dst_points)

    # Calculate homography matrix (transformation between source and destination)
    # ordering is necessary because the src and dst poitns must match their order
    homography, mask = cv.findHomography(
        order_points(src_points), order_points(dst_points)
    )

    # Get image size
    h, w, _ = img.shape

    # Reproject the image using homography
    warped_img = cv.warpPerspective(img, homography, (w, h))

    return warped_img

def get_corner_dots(img: np.ndarray, min_area: float = 60, max_area: float = 300):
    """
    img is
    """
    # Convert the image to grayscale if needed
    color_image = img
    if img.shape[2] != 1:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # inverted_image = cv2.bitwise_not(thresh, 255)
    # cv2.imshow("Image with Dots Marked", img)
    # cv2.waitKey(0)  # Wait for a keypress to close the window
    # cv2.destroyAllWindows()

    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area  # Adjust based on expected dot size
    params.filterByCircularity = False
    # params.minCircularity = 0.3  # Adjust based on how circular the dots are
    detector = cv.SimpleBlobDetector_create(params)

    thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    
    keypoints = detector.detect(thresh)
    if not keypoints:
        return []
    # print(keypoints)
    corners = [kp.pt for kp in keypoints]
    for x, y in corners:
        cv.circle(thresh, (int(x), int(y)), 20, (0, 255, 0), 3)  
    cv.imshow("Image with Dots Marked", thresh)
    cv.waitKey(0)  # Wait for a keypress to close the window
    cv.destroyAllWindows()
    return order_points(corners)

def resize(img: np.ndarray, target_height: int = 500) -> np.ndarray:
    
    """
    resize an image to have height = target_height
    """
    conversion_factor = target_height / img.shape[0]
    return cv.resize(
        img,
        (int(img.shape[1] * conversion_factor), int(img.shape[0] * conversion_factor)),
    )
    
def get_spot(img: np.ndarray):
    """
    Given an image, compute the coordinates of the spot in the middle
    """
    if img.shape[2] != 1:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    thresh = cv.threshold(img, 220, 255, cv.THRESH_BINARY)[1]
    # cv2.imshow("Middle blob thresh", thresh)
    # cv2.waitKey(0)  # Wait for a keypress to close the window
    # cv2.destroyAllWindows()

    cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # inverted_image = cv2.bitwise_not(thresh, 255)
    cnt_centres = []
    for cnt in cnts:
        # Get moments
        M = cv.moments(cnt)
        # Calculate center coordinates
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Print center coordinates
        # print(f"Center of contour: ({cX}, {cY})")

        cnt_centres.append((cX, cY))

    return cnt_centres

def compute_error_vector(img: np.ndarray) -> np.ndarray:
    """
    Given an image, compute an error vector in meters (x, y)
    of the spot centroid vs the plate center
    """
    # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # img = img[630:2300, 1100:3000, :]  # hardcoded crop region.
    # print(img.shape)
    img = resize(img)
    corner_dots = get_corner_dots(img, 60, 300)
    if len(corner_dots) != 4:
        raise RuntimeError("can't find dots")
    for x, y in corner_dots:
        cv.circle(img, (int(x), int(y)), 20, (0, 255, 0), 3)  # Draw green circle
    img = reproject_to_square(img, corner_dots)
    corner_dots_2 = np.array(get_corner_dots(img, 60, 300)[-1])
    center = corner_dots_2 / 2  # assume it only picked up bottom right corner
    # dst_height = max(point[1] for point in points) - min(point[1] for point in points)
    cv.circle(
        img, (int(center[0]), int(center[1])), 10, (255, 0, 0), 3
    )  # Draw red circle
    # cv2.imshow("Image with Dots Marked", img)
    # cv2.waitKey(0)  # Wait for a keypress to close the window
    # cv2.destroyAllWindows()
    # print(blobs)
    spots = get_spot(img)
    for x, y in spots:
        cv.circle(img, (int(x), int(y)), 50, (0, 0, 255), 3)  # Draw red circle

img = resize(img, target_height=500)
import sys
corners = get_corner_dots(img, min_area=20, max_area=1000)

cv.imshow('Image dot', img)
print(corners)
cv.waitKey(0)
sys.exit(0)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
gray = np.float32(gray)

# find light spot
_, thresholded = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)

# Find contours
contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if area > 100:  # Adjust this threshold as needed
        cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
        # cv.putText(img, f"Spot {i+1}", (contour[0][0][0], contour[0][0][1]),
        #             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
if contours:
    largest_contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(largest_contour)
    if area > 100:
        cv.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            # Calculate x, y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Print the center coordinates
            print(f"Center of Light Spot: ({cX}, {cY})")
            # Draw the center of the spot
            cv.circle(img, (cX, cY), 5, (255, 0, 0), -1)
            # Label the spot
            cv.putText(img, "Light Spot", (cX, cY),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("Light Spot has zero division error in moments calculation.")
            
else:
    print("No light spots detected.")
        


# Detect corners using cv.cornerHarris
corners = cv.cornerHarris(gray, 2, 3, 0.05)
corners = cv.dilate(corners, None)

print(corners)

# Mark corners on the image
img[corners > 0.01 * corners.max()] = [0, 0, 255]

# Find the center of the image
height, width = img.shape[:2]
center = (width // 2, height // 2)
print(f'The center coordinates are {center}')

# Mark the center with a red dot
cv.circle(img, center, 5, (0, 0, 255), -1)

# pixels_per_cm = 10 

# distance_pixels = ((cX - center[0]) ** 2 + (cY - center[1]) ** 2) ** 0.5

# # Convert the distance to centimeters
# distance_cm = distance_pixels / pixels_per_cm

# print(f"Distance from the center of the light spot to the center of the image: {distance_cm:.2f} cm")

# new
physical_width_cm = 100  # in centimeters
physical_height_cm = 100  # in centimeters

# Image dimensions in pixels
height, width = img.shape[:2]

# Assuming the physical size is the same for width and height, calculate pixels_per_cm
# You can use either width or height since the image is square
pixels_per_cm = width / physical_width_cm  # or height / physical_height_cm

# Calculate the distance in pixels between the center of the light spot and the center of the image
distance_pixels = ((cX - center[0]) ** 2 + (cY - center[1]) ** 2) ** 0.5

# Convert the distance to centimeters
distance_cm = distance_pixels / pixels_per_cm

print(f"Distance from the center of the light spot to the center of the image: {distance_cm:.2f} cm")

# Display image with corners and center marked
cv.imshow('Image with Corners and Center', img)
cv.waitKey(0)

# Save the image with detected corners and center marked
cv.imwrite(path, img)
print(f"Saved image with corners and center to {path}")






