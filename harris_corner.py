from __future__ import print_function, division
import numpy
import cv2
from scipy.ndimage import maximum_filter


if __name__ == '__main__':
    file_name = 'data/building.png'
    block_size = 2
    aperture_size = 3
    k = 0.04

    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = numpy.float32(gray)

    # OpenCV
    dst = cv2.cornerHarris(gray, block_size, aperture_size, k)

    # Dilate the result to make the corner bigger for display
    dst = cv2.dilate(dst, None)
    opencv_img = img.copy()
    opencv_img[dst > dst.max() * 0.01] = [0, 0, 255]

    # Mine
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=aperture_size)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=aperture_size)

    A = Ix * Ix
    B = Iy * Iy
    C = Ix * Iy
    A = cv2.GaussianBlur(A, ksize=(0, 0), sigmaX=1, sigmaY=1)
    B = cv2.GaussianBlur(B, ksize=(0, 0), sigmaX=1, sigmaY=1)
    C = cv2.GaussianBlur(C, ksize=(0, 0), sigmaX=1, sigmaY=1)

    R = (A * B - (C ** 2)) - k * ((A + B) ** 2)

    # Surpress non local maxima
    threshold = R.max() * 0.01
    local_maxima = numpy.where(
        maximum_filter(R, size=(block_size, block_size)) == R, R, 0)

    local_maxima = cv2.dilate(local_maxima, None)
    mine_img = img.copy()
    mine_img[local_maxima > threshold] = [0, 0, 255]

    cv2.imshow('Opencv harris corner', opencv_img)
    cv2.imshow('My harris corner', mine_img)
    cv2.waitKey()
