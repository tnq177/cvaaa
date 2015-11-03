from __future__ import print_function, division
import numpy
import cv2

img = cv2.imread('../data/noise_1.png', 0)
cv2.imshow('img', img)

fourier_transformed = numpy.fft.fft2(img)
inverse_fourier_transformed = numpy.fft.ifft2(fourier_transformed)
inverse_fourier_transformed = numpy.uint8(inverse_fourier_transformed)

f = numpy.fft.fftshift(fourier_transformed)
magnitude = numpy.uint8(20 * numpy.log(numpy.absolute(f) + 1))


diff = numpy.uint8(img - inverse_fourier_transformed)
cv2.imshow('magnitude', magnitude)
cv2.waitKey()
