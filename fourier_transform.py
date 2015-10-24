from __future__ import print_function, division
import numpy
import cv2

def get_plottable_fft(x):
    mag = numpy.absolute(x)
    mag *= (255.0 / mag.max())
    return mag.astype(numpy.uint8)

# bbc = cv2.imread('./data/BBC_grey_testcard.png', 0)
# bbc_fft = numpy.fft.fft2(bbc)
# bbc_ifft = numpy.fft.ifft2(bbc_fft)
# bbc_ifft = numpy.abs(bbc_ifft).astype(numpy.uint8)

# bbc_fft_centered = numpy.fft.fftshift(bbc_fft)

# cv2.imshow('bbc', bbc)
# cv2.imshow('inverse fft of bbc', bbc_ifft)
# cv2.imshow('fft', get_plottable_fft(bbc_fft))
# cv2.imshow('centered fft', get_plottable_fft(bbc_fft_centered))
# cv2.waitKey()

# # Fourier transform of Cosine if average of two impulses at f & -f, with f = frequency of cosine function
# cosine_x = numpy.zeros((500, 500), numpy.float)
# cosine_y = numpy.zeros((500, 500), numpy.float)

# for i in xrange(500):
#     cosine_x[:, i].fill(255 * numpy.cos(100 * i))
#     cosine_y[i].fill(255 * numpy.cos(100 * i))

# cosine_x_fft_centered = numpy.fft.fftshift(numpy.fft.fft2(cosine_x))
# cosine_y_fft_centered = numpy.fft.fftshift(numpy.fft.fft2(cosine_y))

# cosine_x[cosine_x < 0] = 0
# cosine_y[cosine_y < 0] = 0
# cv2.imshow('cosine x', cosine_x.astype(numpy.uint8))
# cv2.imshow('cosine y', cosine_y.astype(numpy.uint8))
# cv2.imshow('cosine x fft', get_plottable_fft(cosine_x_fft_centered))
# cv2.imshow('cosine y fft', get_plottable_fft(cosine_y_fft_centered))
# cv2.waitKey()

# If fft is an impulse, it should result in a sinusoidal along x & y
img = numpy.zeros((500, 500), numpy.float)

# Very stupid
for x in xrange(500):
    for y in xrange(500):
        angle = numpy.pi * 3 / 4 - numpy.arctan2(y, x)
        r = numpy.sqrt(x ** 2 + y ** 2) * numpy.sin(angle) / (numpy.sin(numpy.pi / 4) * numpy.sqrt(2))
        img[x, y] = numpy.cos(100 * r)

fft_shift = numpy.fft.fftshift(numpy.fft.fft2(img))
cv2.imshow('img', img)
cv2.imshow('fft', get_plottable_fft(fft_shift))
cv2.waitKey()
