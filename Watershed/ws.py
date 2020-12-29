# import the necessary packages
from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils
import numpy as np
import cv2

IMG_FILE = "coins.jpg"

def pre_ws():
	# load the image and perform pyramid mean shift filtering
	# to aid the thresholding step
	image = cv2.imread(IMG_FILE)
	shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
	#cv2.imshow("Input", image)
	#cv2.imwrite("out.jpg", image)

	# convert the mean shift image to grayscale, then apply
	# Otsu's thresholding
	gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv2.imwrite("thresh.jpg", thresh)

	#watershed
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,
		labels=thresh)
	# perform a connected component analysis on the local peaks,
	# using 8-connectivity, then appy the Watershed algorithm
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)
	print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
	print("Labels", np.unique(labels), "Markers", markers, "EDT", -D)
	#cv2.imwrite("wsout.jpg", image)
	#cv2.waitKey(0)
	#label the 1 marked values in markers uni
	return (label_markers, -D)


def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

