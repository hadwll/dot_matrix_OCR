## This script is for capturing images of the box from the
## conveyor line.

import cv2
from gpiozero import Button,LED
from time import sleep
from datetime import datetime


button = Button(2)
red = LED(26)
state = False

def set_res(aspect,cam):
    if aspect == 1:
        width = 1280
        height = 720
    elif aspect == 2:
        width = 1920
        height = 1080
    else:
        aspect =1
    cam.set(3,width) #Setting webcam's image width
    cam.set(4,height) #Setting webcam' image height
    return cam

def capture(button, cam, path, datetime):
    state = False
    ret, image = cam.read()
    cam.release()
    state =True
    if ret:
        cv2.imwrite(path, image)
        print('Photo saved')
        print('.')
        cam.release()
    return (image)
    
## Program loop
while True:
    if button.is_pressed and state == False:
        cam = cv2.VideoCapture(0)
        set_res(2,cam)
        timestamp = datetime.now().isoformat()
        path = ('/home/pi/Scripts/Box_images/%s.jpg' % timestamp)
        image = capture(button, cam, path, datetime)
        state =True
        #cv2.imshow('Train_shot',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    if not button.is_pressed:
        state = False   
    
    
    
            
            

    

