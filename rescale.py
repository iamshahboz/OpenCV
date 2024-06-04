import cv2 as cv 


'''
this function will resize the video we have
'''
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimentions = (width, height)
    
    return cv.resize(frame, dimentions, interpolation=cv.INTER_AREA)


# capture = cv.VideoCapture(0)


capture = cv.VideoCapture('Videos/bird.mp4')

# the video is shown frame by frame 
while True: 
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame)
    
    
    cv.imshow('Video', frame)
    cv.imshow('Video resized', frame_resized)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break 
    
capture.release()
cv.destroyAllWindows()
    