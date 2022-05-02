import cv2
import numpy as np

img = cv2.imread('filename.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharped_img = cv2.filter2D(img, -1, sharpen_filter)

cv2.imshow('sharp',sharped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


sharped_img=cv2.cvtColor(sharped_img,cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(sharped_img,(5,5),0)
_, th2 = cv2.threshold(sharped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(th2)

cv2.imshow('thresholded',inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()

roi = inverted[50:75, 320:490]


cv2.imshow('ROI',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

##dilate and erode to remove the dots from the image.
kernel = np.ones((2,2),np.uint8)
dilated =cv2.dilate(roi,kernel,iterations=2)
eroded = cv2.erode(dilated,(5,5),iterations=2)

cv2.imshow("pre-processed", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

