


from keras.models import load_model
import keras
import cv2
import numpy as np
from skimage.filters import threshold_local
import os


def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass
    
def pre_process(image, inv = False):
    """Uses OTSU binarization on an image"""
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
        pass
    
    if inv == False:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(th2, (32,32), interpolation = cv2.INTER_AREA)
    return resized    


classifier = load_model(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\Trained_Models\ocr.h5")
region = [(160, 300), (640, 290)]


img = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\box_ocr\box1.jpg",0)
orig_img = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\box_ocr\box1.jpg")
dim = (800 , 600)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
orig_img= cv2.resize(orig_img, dim, interpolation = cv2.INTER_AREA)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img= img[300:345, 160:400]

_, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(th2)
cv2.imshow("inverted", inverted)
cv2.waitKey(0)

##dilate and erode to remove the dots from the image.
kernel = np.ones((2,2),np.uint8)
dilated =cv2.dilate(inverted,kernel,iterations=1)
eroded = cv2.erode(dilated,(5,5),iterations=1)

cv2.imshow("dilated and eroded", eroded)
cv2.waitKey(0)


blurred = cv2.blur(eroded, (2,2))

cv2.imshow("blured", blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("edged", edged)
cv2.waitKey(0)


# Find Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#Sort out contours left to right by using their x cordinates
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:11] #the number of expected characters
contours = sorted(contours, key = x_cord_contour, reverse = False)
 
# Create empty array to store entire number
full_number = []



# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)    
    if w >= 5 and h >= 10:
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 20, 255,cv2.THRESH_BINARY_INV)
        cv2.imshow("ROI1", roi)
        roi_otsu = pre_process(roi, True)
        cv2.imshow("ROI2", roi_otsu)
        roi_otsu = cv2.cvtColor(roi_otsu, cv2.COLOR_GRAY2RGB)
        roi_otsu = keras.preprocessing.image.img_to_array(roi_otsu)
        roi_otsu = roi_otsu * 1./255
        roi_otsu = np.expand_dims(roi_otsu, axis=0)
        image = np.vstack([roi_otsu])
        label = str(classifier.predict_classes(image, batch_size = 10))[1]
        full_number.append(label)
        (x, y, w, h) = (x+region[0][0], y+region[0][1], w, h)
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(orig_img, label, (x , y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("image", orig_img)
        cv2.waitKey(0) 
        
cv2.destroyAllWindows()

print(full_number)