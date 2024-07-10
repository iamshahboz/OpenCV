import numpy as np
import cv2 as cv
import random

def capture_and_save():
    capture = cv.VideoCapture(0)

    while True:
        # randomly generating path, for not replacing existing file(with the same name)
        path = f'captures/image_{random.randint(1,100)}.png'

        isTrue, frame = capture.read()
        cv.imshow('Video', frame)

        key = cv.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 13: # ASCII value for Enter key is 13
            cv.imwrite(path, frame)
            img = cv.imread(path)

            # Convert the image to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = np.float32(gray)

            # Detect corners using cv.cornerHarris
            corners = cv.cornerHarris(gray, 2, 3, 0.05)
            corners = cv.dilate(corners, None)

            # Mark corners on the image
            img[corners > 0.01 * corners.max()] = [0, 0, 255]

            # Display image with corners
            cv.imshow('Image with Corners', img)
            cv.waitKey(0)

            # Save the image with detected corners
            cv.imwrite(path, img)
            print(f"Saved image with corners to {path}")

            break

    capture.release()
    cv.destroyAllWindows()
    
    return path

# Call the function to run the code
saved_image_path = capture_and_save()
print(saved_image_path)