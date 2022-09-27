#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:38:35 2019

@author: rebeccarojansky
"""
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
#import matplotlib 
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#from sklearn.cluster import KMeans
#import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import argparse
import cv2
import csv
#from numpy import empty
import glob
#import itertools
import PySimpleGUI as sg
import os
import sys
import easygui
from sklearn.mixture import GaussianMixture as GMM
gmm_labels2 = []

#Initialize midpoints for later contour calculation
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Define color filters for red, pink, white, and grey
#lower_red = np.array([0,0,0])
#upper_red = np.array([30,255,255]) 
lower_pink = np.array([150,0,0])
upper_pink = np.array([210,255,255]) 
##lower_white = np.array([0,0,177])
##upper_white = np.array([180,16,255]) 
#lower_grey = np.array([0,0,86])
#upper_grey = np.array([351,30,255]) 
#lower_blue = np.array([240,0,0])
#upper_blue = np.array([255,255,255])


#Initialize list variables that will hold data
name_list = []
number_list = []
dimA_list = []
dimB_list = []
area_list = []
ratio_list = []
name_list_2 = []
number_list_2 = []
area_list_2 = []
positivegroup = []

#Load column headers in list variables
name_list.append('filename')
number_list.append('contour number')
dimA_list.append('length')
dimB_list.append('width')
area_list.append('area')
ratio_list.append('ratio')
name_list_2.append('filename')
area_list_2.append('total area')
number_list_2.append('total fragments')

positivegroupfound = False

#pixelsPerMetric = float(input("Please enter the image scale in pixels/micron:")) #0.083
pixelsPerMetric = float(easygui.enterbox("Please enter the image scale in pixels/micron"))

# Read in the images looping over all images with tif extension in the folder containing this script
# Create a list of the filenames stored in "Path"
if len(sys.argv) == 1:
    event, values = sg.Window('Choose Image Directory').Layout([[sg.Text('Folder to open')],
                                                   [sg.In(), sg.FolderBrowse()],
                                                   [sg.CloseButton('Open'), sg.CloseButton('Cancel')]]).Read()
    fname = values[0]
    #print(event, values)
else:
    fname = sys.argv[1]

if not fname:
    sg.Popup("Cancel", "No filename supplied")
    raise SystemExit("Cancelling: no filename supplied")
os.chdir(fname,)
path = glob.glob("*.jpg")
path2 = glob.glob("*.tif")
path3 = glob.glob("*.tiff")
path4 = glob.glob("*.png")
path.extend(path2)
path.extend(path3)
path.extend(path4)
#print ('zero')


cv2_img = []
#for img in path:

    # pull out just the s channel
#    lu=im_hsv[...,0].flatten()
#    plt.hist(lu,256)
#    plt.show()
#    lu=im_hsv[...,1].flatten()
#    plt.hist(lu,256)
#    plt.show()
#    lu=im_hsv[...,2].flatten()
#    plt.hist(lu,256)
#    plt.show()
    
    
#    gmm_model = GMM(n_components=5, covariance_type='tied').fit(img2)
#    gmm_labels = gmm_model.predict(img2)
#    gmm_params = gmm_model.get_params(deep=True)
#    gmm_means = gmm_model.means_
#    gmm_covar = gmm_model.covariances_
    #original_shape = n.shape
    #segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
    #cv2.imwrite("segmentedimage.jpg", segmented)
    
    #cv2_img.append(n)
   
# Looping over all the images create a grayscale image, then blur it slightly, then apply a median filter
b = len(path)
for a in range(0, b):
	currentname = path[a]
	n = cv2.imread(currentname)
	cv2_img.append(n)
    # convert, but this is buggy 
	hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
	#print('0.5')
    
	mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)#
	mask_pink = 255 - mask_pink
	hsv = cv2.bitwise_and(hsv, hsv, mask=mask_pink)
	cv2.imwrite(currentname + '_pinkmask.png', hsv)    
	img2 = n.reshape((-1,3))
#img = scipy.misc.imread("/home/subhradeep/Desktop/test.jpg")
	#array=np.asarray(n)
    #hsv = matplotlib.colors.rgb_to_hsv(array[...,:3])
	#print ('one')
# Normalize the image
#	def normalize(x):
#		return (x - 128.0) / 128
#	cv2_img[a] = normalize(cv2_img[a])
#im = array(Image.open('AquaTermi_lowcontrast.jpg').convert('L'))
#im2,cdf = imtools.histeq(im)
#FILTERING    
    #convert the RGB image to HSV format for color filtering
#	hsv = cv2.cvtColor(cv2_img[a], cv2.COLOR_BGR2HSV)
	#hsv = cv2.GaussianBlur(hsv, (17, 17), 0)
	gmm_model = GMM(n_components=4, covariance_type='tied').fit(img2)
	#print ('two')
	gmm_labels = gmm_model.predict(img2)
	#gmm_params = gmm_model.get_params(deep=True)
	gmm_means = gmm_model.means_
	#gmm_covar = gmm_model.covariances_

	original_shape = n.shape
	segmented = gmm_labels.reshape(original_shape[0], original_shape[1])

	colorlength = len(gmm_means)
	print(currentname)
	print(gmm_means)
	for q in range(0,colorlength):
		mean = gmm_means[q]
		#if positivegroupfound != True:
			#print (mean[0])
		if mean[0] >= 100 and mean[0] <= 120 and mean[1] <= 180 and mean[2] <= 250:# and mean[2] >= 200: #and mean[0] <= 230 and mean[1] >= 150:
			positivegroup.append(q)            
		else:
			if mean[0] >= 130 and mean[0] <= 230 and mean[1] <= 180 and mean[2] <= 250:# and mean[2] >= 200: #and mean[0] <= 230 and mean[1] >= 150:
				positivegroup.append(q)            
	#		positivegroupfound = True        
	#cv2.imwrite("segmentedimage.jpg", segmented)
	#cv2.imshow(gmm_means)
	#print('three')
	labelslength = len(gmm_labels)
	for b in range(0, labelslength):
		gmm_labels2.append(gmm_labels[b])
		for group in positivegroup:          
			if gmm_labels[b] == group:                
				gmm_labels[b] = 255
				#print("group")
				#print(group)
	for b in range(0, labelslength):
		if gmm_labels[b] != 255:
			gmm_labels[b] = 0
	original_shape = n.shape
	segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
	segmented2 = np.uint8(segmented)
	hsv2 = cv2.bitwise_and(hsv, hsv, mask=segmented2)        
	cv2.imwrite(currentname + '_mask.png', hsv2)
	#print("positive groups")
	print(positivegroup)
	positivegroup = []    
    

	Median = cv2.medianBlur(hsv2, 1)	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	Median = cv2.dilate(Median, kernel, iterations=1) 
	Median = cv2.erode(Median, kernel, iterations=4) 
	Median = cv2.dilate(Median, kernel, iterations=5) 

	cv2.imwrite(currentname + '_dilated.png', Median) 

    
#EDGE DETECTION    
    # perform Canny edge detection, then perform a dilation + erosion to
    # close gaps in between object edges *note that the number of iterations of dilation and erosion 
    # will change what clusters together into a single contour
	edged = cv2.Canny(Median, 3, 3)
	edged = cv2.dilate(edged, None, iterations=3)	
	edged = cv2.erode(edged, None, iterations=3)

#OPTIONAL: Show the binary mask after dilation and erosion
	#cv2.imshow('edged',edged)
	#cv2.waitKey(0)    
	#cv2.destroyAllWindows()
    	
    #apply the binary mask to the original image    
	ret,mask =cv2.threshold(edged,127,255,cv2.THRESH_BINARY_INV)
    
#CONTOUR DETECTION
    #Initialize a mask that will be used to remove the small contours
	small = np.ones(hsv.shape[:2], dtype="uint8") * 255
    
    #find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,	cv2.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
	screenCnt = None
 
	if cnts != []:
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable *NOTE: change pixelspermetric 
    #based on the calibration of the microscope and camera
		(cnts, _) = contours.sort_contours(cnts)
	#pixelsPerMetric = 0.083

#FINDING AREA OF CONTOURS    
    #initialize the counters to loop over the contours
	d = 0
	e = 0
	area_sum = 0
    # loop over the contours individually
	for c in cnts:
		d = d+1
		strd = str(d)

    # compute the rotated bounding box of the contour
		orig = cv2_img[a].copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
 
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
		box = perspective.order_points(box)
		#cv2.drawContours(orig, c, -1, (0, 255, 0), 3)
		#cv2.imwrite(currentname + strd + '_cnt.png', orig)
 
	# loop over the original points and draw them
		#for (x, y) in box:
		#	cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
 
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
 
	# draw the midpoints on the image
		#cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		#cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		#cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		#cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
		#cv2.drawContours(orig, cntour, -1, (0, 255, 0), 3)
 
	# draw lines between the midpoints
		#cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		#	(255, 0, 255), 2)
		#cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		#	(255, 0, 255), 2)
    # compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		
    # compute the size of the object in microns (depends on calibration set above)
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
		area = dimA * dimB
		#ratio = dB/dA	
        #filter out contours with area less than a set value *NOTE: 200000 sqaure microns filters out irrelevant fragments
        # contours larger than this threshold are added to the list variables
		if area >= 200000:
			e=e+1
			name_list.append(currentname)
			number_list.append(d)
			dimA_list.append(dimA)
			dimB_list.append(dimB)
			area_list.append(area)
			#ratio_list.append(ratio)
			area_sum = area_sum + area
# Add small contours to mask for deletion
			cv2.drawContours(small, [c], -1, 0, -1)

#FINAL OUTPUT
    #OPTIONAL: Output the dimensions of the current bounding box in text form for troubleshooting
			#print(currentname, ",", dimA, ",", dimB, ",") 

	# draw the object sizes on the image
			cv2.putText(orig, "{:.1f}um".format(dimA),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(orig, "{:.1f}um".format(dimB),
				(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
    # Save the final image to a file identified by the original image name and the ratio  of the bouding box
			newfilename = currentname
			newfilename = "%s %s" % (currentname, area_sum)
	
		#if area <1000000: 
		#	cv2.drawContours(small, [c], -1, 0, -1)
    # Save the file as a .png. You can change this extension here.
			#cv2.imwrite(newfilename + '.png', orig)
        
    #create lists containing the file names, number of contours, and the summed area of all contours    
	name_list_2.append(currentname)
	number_list_2.append(e)
	area_list_2.append(area_sum)
	ret,smaller =cv2.threshold(small,127,255,cv2.THRESH_BINARY_INV)
    #Apply smaller mask to original image
	if cnts != []:
		orig2 = cv2.bitwise_and(orig, orig, mask=smaller)
	#cv2.imshow("SmallMask", small)
	#cv2.imshow("AfterSmallMask", orig)
	#cv2.waitKey(0)
		cnts2 = cv2.findContours(smaller.copy(), cv2.RETR_LIST,	cv2.CHAIN_APPROX_SIMPLE)
		cnts2 = imutils.grab_contours(cnts2)
		cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[:100]
		screenCnt = None
		cv2.drawContours(orig, cnts2, -1, (0, 255, 255), 5)
		cv2.imwrite(currentname + '_contours.png', orig)
		#cv2.imwrite(currentname + '_Kmeans.png', res2)
		#cv2.imwrite(currentname + '_Mask.png', res3)
		#cv2.imwrite(currentname + '_Kmeans2.png', res5)
		#cv2.imwrite(currentname + '_hsv.png', hsv)
		#cv2.imwrite(currentname + '_Median.png', Median)
		positivegroupfound = False
print ('round 1')
#initialize counters to fill csv files
i = 0    
l = 0
j = len(name_list)
k = len(name_list_2)

#write csv files
with open('dimension_file.csv', mode='w') as dimension_file:
	dimension_writer = csv.writer(dimension_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(0, j):
			row = [name_list[i], number_list[i], dimA_list[i], dimB_list[i], area_list[i]]
			dimension_writer.writerow(row)   
with open('size_file.csv', mode='w') as size_file:
	size_writer = csv.writer(size_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for l in range(0, k):
			row2 = [name_list_2[l], number_list_2[l], area_list_2[l]]
			size_writer.writerow(row2)      