#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:38:42 2021

@author: hunterstuckey


"""

import numpy as np
import imutils
import cv2
import math


##############################################################################
#
#    Ellipse Detection Function
#
##############################################################################

def detect_ellipse(image, contours, edges):
    # size of image
    Width_dimension  = 1620
    Height_dimension = 1080
    # set the parameters for ellipse detection
    min_Contour_Size      = 50  # if this is too large, i can't detect it far away, too small noise gets through
    min_Axis_Size         = 35  # if this is too large, i can't detect it far away. if it's too small noisy ellipses get through
    max_Axis_Size         = 700
    max_Axis_Ratio        = 30
    min_Axis_Ratio        = .02
    small_contour_overlap = .55
    small_ellipse_overlap = .20
    min_angle             = 60
    max_angle             = 120

    # create empty list for ellipses
    ellipses_detected = []
    ellipses_contours = []
    
    # iterate through each contour
    for (i, c) in enumerate(contours):
        
        
        # get the contour pixel count
        c_pixels = np.count_nonzero(c)
        
        # skip this contour if it is too small
        if c_pixels < min_Contour_Size:
            continue

        # reject contours that are near the edge of the image
        x,y,w,h = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or w >= Width_dimension - 2 or h >= Height_dimension - 2:
            continue

        # fill the ellipse
        ellipse = cv2.fitEllipse(c)
        (xc,yc),(smallAxis,largeAxis),angle = ellipse
        
        # reject unreasonable dimensions
        if largeAxis > max_Axis_Size:
            continue
        if smallAxis < min_Axis_Size:
            continue
        
        # reject unreasonable ratio
        axisRatio =  largeAxis / smallAxis
        if axisRatio > max_Axis_Ratio:
            continue
        if axisRatio < min_Axis_Ratio:
            continue

        print(angle)

        # reject if the angle is too large/vertical, the drone shouldn't be stable in those states
        # it looks like it calculates angle's origin from the vertical axis, not horizontal
        if angle < min_angle or angle > max_angle:
            continue

        # make masks for contour/ellipse
        mask_contour   = cv2.drawContours(np.zeros(shape=edges.shape,dtype=np.uint8), [contours[i]], -1, 255, 1)
        mask_ellipse   = cv2.ellipse(np.zeros(image.shape,dtype=np.uint8),ellipse,(0,255,0), 2)
        mask_ellipse   = cv2.cvtColor(mask_ellipse, cv2.COLOR_BGR2GRAY)
        ellipse_pixels = np.count_nonzero(mask_ellipse)
        
        # find the contour overlap with the filled ellipse
        contour_overlap = np.count_nonzero(np.logical_and(mask_contour,mask_ellipse))/c_pixels
    
        # if the overlap is too small reject
        if contour_overlap < small_contour_overlap:
            continue
        
        # find the ellipse overlap with the edge map
        ellipse_overlap = np.count_nonzero(np.logical_and(mask_ellipse,edges))/ellipse_pixels
        
        # if the overlap is too small reject
        if ellipse_overlap < small_ellipse_overlap:
            continue 
        
        # add ellipse that passed all conditions
        ellipses_detected.append(ellipse)
        # add associated contour
        ellipses_contours.append(c)

        
        
    return ellipses_detected,ellipses_contours



##############################################################################
#
#   Spherical To Cartesian Function
#
##############################################################################

def sph2cart(phi,theta,rho):
    y = rho * np.sin( theta )
    x = rho * np.sin( phi   )
    z = rho * np.cos( theta )
    return [x,y,z]

    

##############################################################################
#
#   Optical Localization Function
#
##############################################################################


def find_location(ellipse_contour):
    # find the distance of the detected ellipse in this function

    # parameters for the camera, my camera outputs 3:2. So 1620x1080
    Width_dimension  = 1620
    Height_dimension = 1080
    VFOV             = (58.397 * np.pi) / 180
    HFOV             = (58.397 * np.pi) / 180
    VFOV_constant    = np.tan(VFOV/2)/Height_dimension
    HFOV_constant    = np.tan(HFOV/2)/Width_dimension
    imcenter_y       = Height_dimension/2
    imcenter_x       = Width_dimension/2
    ellipse_diameter = .07 # in meters

    # find xl,xr and calculate the center
    xl     = (ellipse_contour[ellipse_contour[:, :, 0].argmin()][0])
    xr     = (ellipse_contour[ellipse_contour[:, :, 0].argmax()][0])
    center = (xl+xr) / 2

    # center each of the points
    Xo = (center[0] - imcenter_x)
    Yo = -1.0 * (center[1] - imcenter_y)

    # calculations for polar coordinates
    theta = math.degrees(np.arctan(math.radians(Yo*2*VFOV_constant)))
    phi   = math.degrees(np.arctan(math.radians(Xo*2*HFOV_constant)))
    rho   = (ellipse_diameter/2) * (abs(Xo/(center[0]-xr[0])))*np.sqrt((1+(1/pow(np.tan(phi),2))))

    # convert to cartesian coordinates
    xyz   = sph2cart(phi,theta,rho)

    return xyz,center



##############################################################################
#
#   Main/Video Streaming
#
##############################################################################

# start camera stream
cap = cv2.VideoCapture(0)

# settings for font
font      = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .48
color     = (255, 0, 0)
thickness = 1

while True:
    
    #take one frame of the video
    _, frame = cap.read()

    # color segmentation
    hsv_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    light_orange = (3, 175, 160) # previous value 1,190, 200
    dark_orange  = (23, 255, 255) # previous value 18,255,255
    
    # bitwiseand over the original image with mask
    mask            = cv2.inRange(hsv_frame, light_orange, dark_orange)
    frame_segmented = cv2.bitwise_and(frame, frame, mask=mask)

    accumEdged = np.zeros(frame_segmented.shape[:2], dtype="uint8")
    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(frame_segmented):
        # blur the channel, extract edges from it, and accumulate the set
	    # of edges for the image
	    chan       = cv2.medianBlur(chan, 5)
	    edged      = cv2.Canny(chan, 25, 200)
	    accumEdged = cv2.bitwise_or(accumEdged, edged)
        

    # add dilation here
    kernel     = np.ones((7,7), np.uint8)
    accumEdged = cv2.dilate(accumEdged, kernel, iterations=1)

    # erosion for cleaning up noise
    kernel     = np.ones((5,5),np.uint8)
    accumEdged = cv2.erode(accumEdged, kernel, iterations=1)
        
    # find contours in the accumulated image, keeping only the largest
    cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
   
    # run ellipse detection
    ellipses,contours = detect_ellipse(frame, cnts, accumEdged)
    
    # if there were ellipses detected
    if len(ellipses) > 0:
        # draw all the ellipses detected
        for i in range(len(ellipses)):
            # draw the ellipse onto the image
            frame = cv2.ellipse(frame, ellipses[i], (255, 0, 0), 10)
            # find location and draw location data next to it
            location,center = find_location(contours[i])
            # write the location data inside the ellipse
            frame = cv2.putText(frame,
            'x:' + str(round(location[0],2)) + '\ny:' + str(round(location[1],2)) + '\nz:' + str(round(location[2],2)),
             org=(int(center[0]),int(center[1])), fontFace=font, fontScale=fontScale, color=color, thickness=thickness)

    
    # show the new image
    cv2.imshow("Detected Ellipses",frame)
     
    # show the accumulated edge map
    cv2.imshow("Edge Map", accumEdged)

    # stop program if esc key is pressed
    key = cv2.waitKey(1)

    if key == 27:
        break
    
    
    
    

