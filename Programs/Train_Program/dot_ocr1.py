## This  code is for the capture and recogniton of dot matrix print.
##
##
#####################################################################
#########################Import Libraries############################
#####################################################################
import cv2
import numpy as np 
import random
import os
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imutils
from keras.models import load_model
import visualkeras

##not used may delete
from skimage.filters import threshold_local
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.layers import Activation, Dropout, Flatten, Dense
from scipy.ndimage import convolve



#####################################################################
#########################Functions###################################
#####################################################################


## Function creates directories for each of the characters 1-10 for the 
## train and test cases
def makedir(directory):
    """Creates a new directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None, 0
    
## The follwoing functions  are for data augmentation 
def DigitAugmentation(frame, dim = 32):
    #Randomly alters the image using noise, pixelation and streching image functions
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    random_num = np.random.randint(0,9)
 
    if (random_num % 2 == 0):
        frame = add_noise(frame)
    if(random_num % 3 == 0):
        frame = pixelate(frame)
    if(random_num % 2 == 0):
        frame = stretch(frame)
    frame = cv2.resize(frame, (dim, dim), interpolation = cv2.INTER_AREA)
 
    return frame 
 
def add_noise(image):
    #Addings noise to image
    # original value prob = random.uniform(0.01, 0.05)
    prob = random.uniform(0.01, 0.03)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy
 
def pixelate(image):
    #Pixelates an image by reducing the resolution then upscaling it
    #original value dim = np.random.randint(8,12)
    dim = np.random.randint(14,16)
    image = cv2.resize(image, (dim, dim), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (16, 16), interpolation = cv2.INTER_AREA)
    return image
 
def stretch(image):
    #Randomly applies different degrees of stretch to image
    #ran = np.random.randint(0,3)*2
    ran = np.random.randint(0,3)*2
    if np.random.randint(0,2) == 0:
        frame = cv2.resize(image, (32, ran+32), interpolation = cv2.INTER_AREA)
        return frame[int(ran/2):int(ran+32)-int(ran/2), 0:32]
    else:
        frame = cv2.resize(image, (ran+32, 32), interpolation = cv2.INTER_AREA)
        return frame[0:32, int(ran/2):int(ran+32)-int(ran/2)]

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
 
def doc_Scan(image):
    orig_height, orig_width = image.shape[:2]
    ratio = image.shape[0] / 500.0
 
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    orig_height, orig_width = image.shape[:2]
    Original_Area = orig_height * orig_width
    
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
 
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    # show the original image and the edge detected image
 
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    contours, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
    # loop over the contours
    for c in contours:
 
        # approximate the contour
        area = cv2.contourArea(c)
        if area < (Original_Area/3):
            print("Error Image Invalid")
            return("ERROR")
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
 
    # show the contour (outline) of the piece of paper
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
 
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    cv2.resize(warped, (640,403), interpolation = cv2.INTER_AREA)
    cv2.imwrite("credit_card_color.jpg", warped)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = warped.astype("uint8") * 255
    cv2.imshow("Extracted Credit Card", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped


def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass
    
#####################################################################
#####################################################################
#####################################################################










#####################################################################
########create an image dataset in order to train the model.#########
#####################################################################

##apply  pre procesing in order to join the dots present in dot matrix print

cc1 = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\train_pic_3.png", 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(th2)


##dilate and erode to remove the dots from the image.
#kernel = np.ones((3,3),np.uint8)
#dilated =cv2.dilate(inverted,kernel,iterations=4)
#cv2.imshow("Dilated", dilated)

#eroded = cv2.erode(dilated,(5,5),iterations=10)
cv2.imshow("inverted", inverted)



cc1= inverted[:,:]
cv2.imshow("Digits 2 Thresholded", cc1)
cv2.waitKey(0)   
cv2.destroyAllWindows()


## create the directories 
for i in range(0,10):
    directory_name = r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\train/" +str(i)
    print(directory_name)
    makedir(directory_name) 
 
for i in range(0,10):
    directory_name = r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\test/"  +str(i)
    print(directory_name)
    makedir(directory_name)





#### The following block of code  creates 2000 test images from the train image ####


# This is the coordinates of the region enclosing  the first digit
# This is preset and was done manually based on this specific image
region = [(0, 0), (40, 120)]



# Assigns values to each region for ease of interpretation
top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]
 

cv2.destroyAllWindows()

for i in range(0,10):   
    # We jump the next digit each time we loop
    if i > 0:
        top_left_x = top_left_x + 43
        bottom_right_x = bottom_right_x + 43
 
    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Train Digit - ", str(i))
    # We create 200 versions of each image for our dataset
    for j in range(0,2000):
        roi2 = DigitAugmentation(roi)
        roi_otsu = pre_process(roi2, inv = True)
        cv2.imwrite(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\train/" +str(i)+"./_1_"+str(j)+".png", roi_otsu)
  
        



#### The following block of code creates 200 test images from the train image ####


cc1 = cv2.imread(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\train_pic_3.png", 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(th2)

##dilate and erode to remove the dots from the image.
#kernel = np.ones((4,4),np.uint8)
#dilated =cv2.dilate(inverted,kernel,iterations=3)
#eroded = cv2.erode(dilated,(5,5),iterations=10)

cc1= inverted

region = [(0, 0), (40, 120)]
 
top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]
 
for i in range(0,10):   
    # We jump the next digit each time we loop
    if i > 0:
        top_left_x = top_left_x + 43
        bottom_right_x = bottom_right_x + 43
 
    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Test Digit - ", str(i))
    # We create 200 versions of each image for our dataset
    for j in range(0,200):
        roi2 = DigitAugmentation(roi)
        roi_otsu = pre_process(roi2, inv = True)
        cv2.imwrite(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\test/"+str(i)+"./_1_"+str(j)+".jpg", roi_otsu)


cv2.destroyAllWindows()

#####################################################################
################### Build deep learning model. ######################
#####################################################################

input_shape = (32, 32, 3)
img_width = 32
img_height = 32
num_classes = 10
nb_train_samples = 20000
nb_validation_samples = 4000
batch_size = 500
epochs = 20


train_data_dir = r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\train/"
validation_data_dir = r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\test/"


# Creating our data generator for our test data
validation_datagen = ImageDataGenerator(
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    rescale = 1./255)
 
# Creating our data generator for our training data
train_datagen = ImageDataGenerator(
      rescale = 1./255,              # normalize pixel values to [0,1]
      rotation_range = 10,           # randomly applies rotations
      width_shift_range = 0.25,       # randomly applies width shifting
      height_shift_range = 0.25,      # randomly applies height shifting
      shear_range=0.5,
      zoom_range=0.5,
      horizontal_flip = False,        # randonly flips the image
      fill_mode = 'nearest')         # uses the fill mode nearest to fill gaps created by the above
 
# Specify criteria about our training data, such as the directory, image size, batch size and type 
# automagically retrieve images and their classes for train and validation sets
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = False)  
  
# create model
model = Sequential()
 
# 2 sets of CRP (Convolution, RELU, Pooling)
model.add(Conv2D(20, (5, 5),
                 padding = "same", 
                 input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
 
model.add(Conv2D(50, (5, 5),
                 padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
 
# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

 
# Softmax (for classification)
model.add(Dense(10))
model.add(Activation("softmax"))
           
model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
    
print(model.summary())
                   
checkpoint = ModelCheckpoint(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\Trained_Models\ocr.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
 
earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)
 
# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]
 
# Note we use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])
 

 
history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
 
model.save(r"C:\Users\hadwl\Documents\University\pervasive computing\Images\OCR_data\Trained_Models\ocr.h5")

from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
#visualkeras.layered_view(model, legend=True, font=font)  # font is optional!
visualkeras.layered_view(model,legend=True, font=font, to_file='output.png')









