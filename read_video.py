import cv2 as cv 




# for video capture 0 means you web camera, 1 first camera plugged and so on 
capture = cv.VideoCapture(0)


# capture = cv.VideoCapture('Videos/bird.mp4')

# the video is shown frame by frame 
while True: 
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break 
    
capture.release()
cv.destroyAllWindows()

 

