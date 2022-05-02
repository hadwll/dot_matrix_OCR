# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:27:12 2022

@author: hadwl
"""

import cv2
import imutils
import numpy as np


def auto_canny(image, sigma = 0.1):
    v = np.median(image)
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(min(255,(1.0+sigma)*v))
    edged=cv2.Canny(image,lower,upper)
    return (edged)
    
image = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\box_ocr\box_2.jpg")
dim = (640,480)
image= cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
original_image = image
cv2.imshow('original', image)
cv2.waitKey(0)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(gray, (3, 3), 0)
#edged = auto_canny(blured)
edged = cv2.Canny(blured, 90, 95)

cv2.imshow("Blured", blured)
cv2.imshow("Edged", edged)


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
cv2.imshow('contoured', original_image)
cv2.waitKey(0)