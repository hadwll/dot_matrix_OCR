## This script will be present on the Rpi

import cv2
import numpy as np
import imutils
import tensorflow.keras
from tensorflow.keras.models import load_model
from datetime import datetime



def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass
    
def pre_process(image, inv = True):
    #Uses OTSU binarization on an image
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
        pass
    
    if inv == True:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(th2, (32,32), interpolation = cv2.INTER_AREA)
    return resized
    

def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect
 
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped


def interpret_date(received_date, max_days):
    
    ##convert to integer format
    received_date = list(map(int,received_date))
    year = received_date[:4]
    str_year = [str(year) for year in year]
    a_year = "".join(str_year)
    int_year = int(a_year)
    
    month =received_date[5:7]
    str_month = [str(month) for month in month]
    a_month = "".join(str_month)
    int_month = int(a_month)
    
    day =  received_date[8:]
    str_day = [str(day) for day in day]
    a_day = "".join(str_day)
    int_day = int(a_day)
    
    
    date = [int_year ,int_month, int_day]
    dt = datetime(*date)
    
    #print(dt)
    #print(datetime.now())
    
    now = datetime.now()
    pack = dt
    delta = now - pack
    
    if abs(delta.days) > max_days:
        return(False)

    else:
        return(True)
    
    
        

image = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\box_ocr\test_02.jpg")
dim = (800,600)
image= cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
original_image = image
#cv2.imshow('original', image)
cv2.waitKey(0)

# convert the image to grayscale, blur it, and find edges
# in the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# Extracting the area were the date numbers are located
roi = gray[320:350, 275:460]

cv2.imshow('ROi', roi)
cv2.waitKey(0)


## greyscale and treshhold the image
_, th2 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(th2)



##dilate and erode to remove the dots from the image.
kernel = np.ones((2,2),np.uint8)
dilated =cv2.dilate(inverted,kernel,iterations=1)
eroded = cv2.erode(dilated,(1,1),iterations=1)

cv2.imshow("pre-processed", inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()


#####################################################################
################# classify the text from the image ##################
#####################################################################


classifier = load_model(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\Trained_Models\ocr.h5")
region = [(275, 319), (600, 440)]


orig_img = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\box_ocr\test_02.jpg")
dim = (800 , 600)
orig_img = cv2.resize(orig_img, dim, interpolation = cv2.INTER_AREA)
img = inverted

 

# Find Contours
contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#Sort out contours left to right by using their x cordinates
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #Change this to 10 date to get all digits
contours = sorted(contours, key = x_cord_contour, reverse = False)
 
# Create empty array to store entire number
full_number = []



# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)    
    if w >= 5 and h >= 10:
        roi = img[y:y + h, x:x + w]
        #ret, roi = cv2.threshold(roi, 20, 255,cv2.THRESH_BINARY_INV)
        cv2.imshow("ROI1", roi)
        roi_otsu = pre_process(roi, True)
        cv2.imshow("ROI2", roi_otsu)
        roi_otsu = cv2.cvtColor(roi_otsu, cv2.COLOR_GRAY2RGB)
        roi_otsu = tensorflow.keras.preprocessing.image.img_to_array(roi_otsu)
        roi_otsu = roi_otsu * 1./255
        roi_otsu = np.expand_dims(roi_otsu, axis=0)
        image = np.vstack([roi_otsu])
        label = str(classifier.predict_classes(image, batch_size = 10))[1]
        full_number.append(label)
        (x, y, w, h) = (x+region[0][0], y+region[0][1], w, h)
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(orig_img, label, (x , y + 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0, 255), 2)
        cv2.imshow("image", orig_img)
        cv2.waitKey(0) 
        
cv2.destroyAllWindows()
print(full_number)
print(interpret_date(full_number, 90))


