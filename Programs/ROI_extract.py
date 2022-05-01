##This code is a test project that attempts to segment a ROI and further zooms
## into the text in  ordrer to prepare it for classification



import cv2
import imutils
import numpy as np

image = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\test.jpg")


orig_height, orig_width = image.shape[:2]
ratio = image.shape[0] / 500.0
 
orig = image.copy()
image = imutils.resize(image, height = 1000)
orig_height, orig_width = image.shape[:2]
Original_Area = orig_height * orig_width


#cv2.imshow("original", image)
cv2.imwrite("display.jpg", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Extracting the area were the credit numbers are located
roi = image[500:531, 590:790]


## greyscale and treshhold the image
roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
_, th2 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(th2)



##dilate and erode to remove the dots from the image.
kernel = np.ones((2,2),np.uint8)
dilated =cv2.dilate(inverted,kernel,iterations=2)
eroded = cv2.erode(dilated,(2,2),iterations=2)

roi =eroded

cv2.imwrite("test2.jpg",roi)

cv2.imshow("pre-processed", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


