import numpy
import cv2

images = ['bottom_dot', 'bottom_left_dot', 'bottom_right_dot', 'center_dot', 'top_dot', 'top_left_dot', 'top_right_dot']

for image_name in images:
    fft = cv2.imread('../data/{0}.png'.format(image_name), 0)
    inverse_fft = numpy.fft.ifft2(fft)
    inverse_fft = numpy.uint8(inverse_fft)

    cv2.imshow('fft', fft)
    cv2.imshow('image (inverse fft)', inverse_fft)
    cv2.waitKey()
