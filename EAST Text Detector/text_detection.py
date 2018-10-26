# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import gc
import math
from scipy import ndimage

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
#image = cv2.imread(args["image"])
image = cv2.imread('images/book-cover-abstract-art.jpeg')
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
#(newW, newH) = (args["width"], args["height"])
(newW, newH) = (640, 640)
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < 0.5:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# draw the bounding box on the image
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
#cv2.imshow("Text Detection", orig)
#cv2.waitKey(0)

del w, blob, confidences, orig, H,W, angle, anglesData, args, cos, end, endX, endY, geometry, h, layerNames, newH, newW, numCols, numRows, offsetX, offsetY, rH, rW, scores, scoresData, sin, start, startX, startY, x, xData0, xData1, xData2, xData3, y

# Extract individual words from the image.
words = []
for i in range(len(boxes)):
    word = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
    words.append(word)  
    
del i
gc.collect()

#-------------------Use sliding windows to get letters----------------------------
for i in range(len(words)):
    cv2.imwrite(str(i) + ".jpg", words[i])

del i

word = words[0]
word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
new_W = int(round(word.shape[1] / 20)) * 20
w = int(word.shape[1] / 20) * 20
# convert each word to grayscale and binarize amd convert height to 20 px.
new_words = []
i = 0
for word in words:
    word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
    word = cv2.threshold(word, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    new_H = 20
    new_W = int(word.shape[1] / 20) * 20
    word = cv2.resize(word, (new_W, new_H))
    new_words.append(word)
    cv2.imwrite(str(i) + ".jpg", word)
    i+= 1

words = new_words
del new_words, new_H, new_W, i 
gc.collect()

# Turn the image into 28x28 pixels with the letter at center.
def process_word_to_mnist_format(word):
    rows, cols = 20, 20
    # Add a padding on all sides to turn into 28 * 28
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    word = np.lib.pad(word,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(word)
    word = shift(word,shiftx,shifty)    
    
    return word
    
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted   

cv2.imwrite('new.jpg', process_word_to_mnist_format(words[0]))

# load the model from disk
from keras.models import load_model

detection_model = load_model('text_detection_model-1.h5')
detection_model.summary()

new = cv2.imread('0.jpg', 0)
alpha_1 = np.lib.pad(new[:, 15:35],(4,4),'constant')
alpha = process_word_to_mnist_format(new[:, 18:38]) # row, column

cv2.imwrite('alpha.jpg', alpha)

def sliding_windows(word, stepSize=2):
    

#---------------------------------------------------------------------------------
# Preprocess a single word to get a grayscale image. 
word = words[1]
gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)[1]
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# Find individual characters using contour finding.
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
 
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)
#---------------------------------------------------------------------------------
# Use cv2.findContours to get all the contours in a word
word = words[1]
word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
_, word = cv2.threshold(word, 127, 255, cv2.THRESH_BINARY)
(W, H) = word.shape
word = cv2.resize(word, (10*H, 10*W))
del W, H
gc.collect()
cv2.imwrite('word.jpg', word)
#gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
#gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
##_,thresh = cv2.threshold(gray,127,255,0)
#_, contours, _ = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
## Draw the found contours on the original image
#cv2.drawContours(word, contours, -1, (0,255,0), 3)
#cv2.imshow("Characters", word)

# Try to isolate letters
im = cv2.imread('word.jpg', 0)
_, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 3)
    
i = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > im.shape[0]/len(contours): # and h > 20:
        cv2.imwrite(str(i) + ".jpg", im[y:y+h, x:x+w])
        i = i + 1
    
import pytesseract
img = cv2.imread('word.jpg', 0)


from PIL import Image
print(pytesseract.image_to_string(img))