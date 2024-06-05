import cv2 as cv
import random
import os

def capture_and_crop():
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
            area = cv.selectROI("select the area", frame)

            # crop the image
            cropped_image = frame[int(area[1]):int(area[1]+area[3]),  
                                  int(area[0]):int(area[0]+area[2])]

            cv.waitKey(0)
            # creating folder for saving cropped images, if exists pass
            try:
                os.makedirs('captures/')
                cv.imwrite(path, cropped_image)
            except FileExistsError:
                cv.imwrite(path, cropped_image)


            break

    capture.release()
    cv.destroyAllWindows()
    
    return cropped_image

# Call the function to run the code
cropped_image = capture_and_crop()
print(cropped_image)
