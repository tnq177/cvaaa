from __future__ import print_function, division
import numpy
import cv2
from common_utils import trackbar_changed_do_nothing


if __name__ == '__main__':
    img = cv2.imread('./data/hypnosis.png', 0)
    gaussian_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=1, sigmaY=1)

    R_0 = cv2.Sobel(gaussian_blur, cv2.CV_32F, 1, 0)
    R_90 = cv2.Sobel(gaussian_blur, cv2.CV_32F, 0, 1)

    cv2.namedWindow('steerable')
    cv2.createTrackbar(
        'angle', 'steerable', 0, 720, trackbar_changed_do_nothing)

    while True:
        angle = numpy.radians(cv2.getTrackbarPos('angle', 'steerable'))
        steered = R_0 * numpy.cos(angle) + R_90 * numpy.sin(angle)
        steered_abs = numpy.uint8(numpy.absolute(steered))

        cv2.imshow('steerable', steered_abs)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
