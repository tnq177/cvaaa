from __future__ import print_function, division
import cv2
import numpy


def gamma_corrected(img, gamma):
    if gamma == 0:
        return numpy.zeros(img.shape)

    lookup_table = build_lookup_table(1 / gamma)

    return cv2.LUT(img, lookup_table)


def build_lookup_table(inv_gamma):
    return numpy.array([((i / 255) ** inv_gamma) * 255 for i in xrange(256)]).astype(numpy.uint8)


def trackbar_changed(x):
    pass

img = cv2.imread('../data/drowning.png', 0)
cv2.namedWindow('img')
cv2.namedWindow('gamma corrected')

cv2.createTrackbar('gamma', 'gamma corrected', 1, 1000, trackbar_changed)
while True:
    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    gamma = cv2.getTrackbarPos('gamma', 'gamma corrected') / 100
    cv2.imshow('gamma corrected', gamma_corrected(img, gamma))

cv2.destroyAllWindows()
