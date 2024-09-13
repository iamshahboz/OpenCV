import cv2 as cv
import random
import os

def capture_and_crop():
    capture = cv.VideoCapture(0)

    while True:
        # Generate a random path for the image capture
        path = f'captures/phone/image_{random.randint(1, 100)}.png'

        isTrue, frame = capture.read()
        cv.imshow('Video', frame)

        key = cv.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 13: # ASCII value for Enter key is 13
            # Save the captured frame to the generated path
            cv.imwrite(path, frame)
            break

    capture.release()
    cv.destroyAllWindows()
    
    return path

# Call the function to run the code
cropped_image = capture_and_crop()
print("Image saved at:", cropped_image)
