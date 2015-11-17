from __future__ import print_function, division
import numpy
import cv2
from os import listdir
from os.path import isfile, join

def get_plottable_fft(x):
    mag = numpy.absolute(x)
    mag *= (255.0 / mag.max())
    return mag.astype(numpy.uint8)

def get_plottable_fft_2(x, scale_factor=20):
    return (scale_factor * numpy.log(numpy.absolute(x) + 1)).astype(numpy.uint8)

def fft_centered(x):
    return numpy.fft.fftshift(numpy.fft.fft2(x))

def trackbar_changed_do_nothing(x):
    pass

def get_images_paths(directory):
    return [join(directory, f) for f in listdir(directory) if isfile(
            join(directory, f)) and f.endswith(('.png', '.jpg', '.JPG'))]
