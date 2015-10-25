from __future__ import print_function, division
import numpy 
import cv2
from common_utils import get_plottable_fft_2, fft_centered, trackbar_changed_do_nothing

img = cv2.imread('./data/lenna.png', 0)

cv2.namedWindow('img')
cv2.namedWindow('blurred')
cv2.namedWindow('img fft')
cv2.namedWindow('blurred fft')
cv2.createTrackbar('ksize', 'blurred fft', 1, 11, trackbar_changed_do_nothing)

while True:
    ksize = cv2.getTrackbarPos('ksize', 'blurred fft')
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0, sigmaY=0)

    img_fft = fft_centered(img)
    blurred_fft = fft_centered(blurred)

    cv2.imshow('img', img)
    cv2.imshow('blurred', blurred)
    cv2.imshow('img fft', get_plottable_fft_2(img_fft, 10))
    cv2.imshow('blurred fft', get_plottable_fft_2(blurred_fft, 10))
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
