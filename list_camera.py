import cv2
from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras(cv2.CAP_MSMF):
    result = {
        'camera_index': camera_info.index,
        'camera_name': camera_info.name
        
    }
    print(result)