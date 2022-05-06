import cv2
import numpy as np
import imutils

image = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\box_ocr\box_14.jpg")
dim = (800,600)
image= cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
original_image = image
cv2.imshow('original', image)
cv2.waitKey(0)








# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
edged = cv2.Canny(image=gray, threshold1=30,threshold2=160)




cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()



# edged is the edge detected image
cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
    
# show the contour (outline) of the piece of paper
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('contoured', image)
cv2.waitKey(0)



cv2.destroyAllWindows()