from __future__ import print_function, division
import numpy
import cv2
from common_utils import trackbar_changed_do_nothing


def get_laplacian_stacks(img):
    lap_stacks = []
    temp_1 = img.copy()
    temp_2 = cv2.pyrDown(temp_1)

    for i in xrange(5):
        temp_2_up = cv2.pyrUp(temp_2)
        lap_stacks.append(cv2.subtract(temp_1, temp_2_up))

        temp_1 = temp_2
        temp_2 = cv2.pyrDown(temp_2)

    lap_stacks.append(temp_1)

    return lap_stacks[::-1]


def merge_two_stacks(stack_1, stack_2, x_percentage):
    result_stack = []
    for img1, img2 in zip(stack_1, stack_2):
        col_number = int(x_percentage * img1.shape[1])
        merged_img = numpy.hstack(
            (img1[:, 0:col_number], img2[:, col_number:]))
        result_stack.append(merged_img)

    return result_stack


def pyr_blending(stack):
    length = len(stack)

    blended = stack[0]
    for i in range(1, length):
        blended = cv2.pyrUp(blended)
        blended = cv2.add(blended, stack[i])

    return blended


if __name__ == '__main__':
    apple = cv2.imread('data/apple.jpg')
    orange = cv2.imread('data/orange.jpg')

    cv2.namedWindow('blended')
    cv2.createTrackbar(
        'horizontal amount', 'blended', 50, 100, trackbar_changed_do_nothing)

    num_cols = apple.shape[1]

    while True:
        # Blending
        apple_stacks = get_laplacian_stacks(apple)
        orange_stacks = get_laplacian_stacks(orange)
        result_stack = merge_two_stacks(apple_stacks, orange_stacks, cv2.getTrackbarPos(
            'horizontal amount', 'blended') / 100)

        blended = pyr_blending(result_stack)
        cv2.imshow('blended', blended)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
