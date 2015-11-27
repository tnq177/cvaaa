from __future__ import print_function, division
import numpy
import cv2
from common_utils import (
    trackbar_changed_do_nothing, 
    get_laplacian_stack, 
    get_gaussian_stack,
    pyr_blending
)


if __name__ == '__main__':
    apple = cv2.imread('data/apple.jpg').astype(numpy.float32)
    orange = cv2.imread('data/orange.jpg').astype(numpy.float32)

    h, w = apple.shape[:2]
    mask_1 = numpy.zeros((h, w, 3), dtype=numpy.float32)
    mask_2 = numpy.zeros((h, w, 3), dtype=numpy.float32)
    mask_3 = numpy.zeros((h, w, 3), dtype=numpy.float32)
    mask_4 = numpy.zeros((h, w, 3), dtype=numpy.float32)
    mask_1[:, :w/2] = 1.0
    mask_2[:, w/2:] = 1.0
    mask_3[:h/2, :] = 1.0
    mask_4[h/2:, :] = 1.0

    apple_stack = get_laplacian_stack(apple)
    orange_stack = get_laplacian_stack(orange)
    mask_stack_1 = get_gaussian_stack(mask_1)
    mask_stack_2 = get_gaussian_stack(mask_2)
    mask_stack_3 = get_gaussian_stack(mask_3)
    mask_stack_4 = get_gaussian_stack(mask_4)

    blended_1 = pyr_blending(apple_stack, orange_stack, mask_stack_1)
    blended_2 = pyr_blending(apple_stack, orange_stack, mask_stack_2)
    blended_3 = pyr_blending(apple_stack, orange_stack, mask_stack_3)
    blended_4 = pyr_blending(apple_stack, orange_stack, mask_stack_4)
    cv2.imshow('blended 1', blended_1)
    cv2.imshow('blended 2', blended_2)
    cv2.imshow('blended 3', blended_3)
    cv2.imshow('blended 4', blended_4)
    cv2.waitKey()