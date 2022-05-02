## script for the capturing images

from gpiozero import Button
from time import sleep
from datetime import datetime
from picamera import PiCamera
from gpiozero import LED

def capture():
    timestamp = datetime.now().isoformat()
    camera.capture('/home/pi/scripts/Box_images/%s.jpg' % timestamp)
    print('took photo')


button = Button(2)
camera = PiCamera()
camera.exposure_mode ='sports'
red = LED(17)
state = False

while True:
    
    if button.is_pressed and state == False: 
        capture()
        red.on()
        sleep(1)
        red.off()
        state =True

    if not button.is_pressed:
        state =False

