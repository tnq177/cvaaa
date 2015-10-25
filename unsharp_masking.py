from __future__ import division
import numpy
import cv2
from common_utils import trackbar_changed_do_nothing

img = cv2.imread('./data/eye.png')
blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
gamma = 0

cv2.namedWindow('img')
cv2.namedWindow('blurred')
cv2.namedWindow('unsharp masking')
cv2.createTrackbar('gamma', 'unsharp masking', 1, 500, trackbar_changed_do_nothing)

while True:
    gamma = cv2.getTrackbarPos('gamma', 'unsharp masking') / 100
    unsharp_masking = cv2.addWeighted(img, 1 + gamma, blurred, -gamma, 0)
    cv2.imshow('img', img)
    cv2.imshow('blurred', blurred)
    cv2.imshow('unsharp masking', unsharp_masking)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    


