import cv2

imageread = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\basic_triangle.jpg")
imagegray= cv2.cvtColor(imageread,cv2.COLOR_BGR2GRAY)
_, imagethreshold = cv2.threshold(imagegray,245,255,cv2.THRESH_BINARY_INV)
imagecontours, hierarchy= cv2.findContours(imagethreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for count in imagecontours:
    epsilon = 0.02 * cv2.arcLength(count, True)
    approximations = cv2.approxPolyDP(count, epsilon, True)
    cv2.drawContours(imageread, [approximations], 0, (0,255,0), 1)
    
#the name of the detected shapes are written on the image
i, j = approximations[0][0] 

if len(approximations) == 3:
    cv2.putText(imageread, "Triangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
elif len(approximations) == 4:
    cv2.putText(imageread, "Rectangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
elif len(approximations) == 5:
    cv2.putText(imageread, "Pentagon", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
elif 6 < len(approximations) < 15:
    cv2.putText(imageread, "Ellipse", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
else:
    cv2.putText(imageread, "Circle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)

#displaying the resulting image as the output on the screen
cv2.imshow("Resulting_image", imageread)
cv2.waitKey(0)