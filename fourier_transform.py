from __future__ import print_function
import numpy
import cv2

def get_plottable_fft(x):
    mag = numpy.absolute(x)
    return 10 * numpy.log(numpy.abs(mag) + 1).astype(numpy.uint8)

bbc = cv2.imread('./data/BBC_grey_testcard.png', 0)
bbc_fft = numpy.fft.fft2(bbc)
bbc_ifft = numpy.fft.ifft2(bbc_fft)
bbc_ifft = numpy.abs(bbc_ifft).astype(numpy.uint8)

bbc_fft_centered = numpy.fft.fftshift(bbc_fft)

cv2.imshow('bbc', bbc)
cv2.imshow('inverse fft of bbc', bbc_ifft)
cv2.imshow('fft', get_plottable_fft(bbc_fft))
cv2.imshow('centered fft', get_plottable_fft(bbc_fft_centered))
cv2.waitKey()

