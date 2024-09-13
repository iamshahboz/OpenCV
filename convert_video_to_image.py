import cv2 
import os 
from datetime import datetime, timedelta
import argparse


def save_frames_from_video(frame_rate:timedelta, save_path:str, source:str=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Open the camera 
    if source:
        video = cv2.VideoCapture(source)
    else:
        video = cv2.VideoCapture(0) 
    
    
    start_time = datetime.now()
    
    while True: 
        # Read video by read() function and it 
        # will extract and  return the frame 
        ret, img = video.read() 
    
        # Put current DateTime on each frame 
        font = cv2.FONT_HERSHEY_PLAIN 
        cv2.putText(img, str(datetime.now()), (20, 40), 
                    font, 2, (255, 255, 255), 2, cv2.LINE_AA) 
    
        
        cv2.imshow('live video', img) 
        
        elapsed_time  = datetime.now() - start_time
        
        if elapsed_time >= timedelta(seconds=args.fps):
            os.makedirs(save_path, exist_ok=True)
            filename = f'{save_path}/{timestamp}.png'
            cv2.imwrite(filename, img)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            start_time = datetime.now()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()

    # Close open windows
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, required=True, help='Set a rate of capture(in seconds)')
    parser.add_argument('--output_dir', type=str, required = True, help='Location to save frames')
    parser.add_argument('--source', type=str, required=False, help='Set the source video')

    args = parser.parse_args()
    fps = timedelta(seconds=args.fps)
    save_frames_from_video(frame_rate=fps,save_path=args.output_dir, source=args.source)
  

  
