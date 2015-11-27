from __future__ import print_function, division
import numpy
import cv2
from common_utils import trackbar_changed_do_nothing


if __name__ == '__main__':
    img = cv2.imread('./data/eye.png')

    cv2.namedWindow('affine')
    cv2.createTrackbar('angle', 'affine', 0, 360, trackbar_changed_do_nothing)
    cv2.createTrackbar('tx', 'affine', 0, 360, trackbar_changed_do_nothing)
    cv2.createTrackbar('ty', 'affine', 0, 360, trackbar_changed_do_nothing)
    cv2.createTrackbar(
        'scale', 'affine', 100, 500, trackbar_changed_do_nothing)

    while True:
        angle = numpy.radians(cv2.getTrackbarPos('angle', 'affine'))
        tx = cv2.getTrackbarPos('tx', 'affine')
        ty = cv2.getTrackbarPos('ty', 'affine')
        scale = cv2.getTrackbarPos('scale', 'affine') / 100

        a = scale * numpy.cos(angle)
        b = scale * numpy.sin(angle)
        M = numpy.array([[a, -b, tx], [b, a, ty]], dtype=numpy.float32)

        transformed_img = cv2.warpAffine(img, M, img.shape[:2][::-1])
        cv2.imshow('affine', transformed_img)

        # Now try to show the whole image in case it's transformed out of image
        # bound
        rows, cols = img.shape[:2]
        # This is the tricky part. For transform, the point is identified as
        # pair of (col, row) instead of (row, col)
        box = numpy.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1],
                           [0, rows - 1]], dtype=numpy.float32).reshape(-1, 1, 2)
        transformed_box = cv2.transform(box, M)
        min_col = min(transformed_box[:, :, 0])[0]
        min_row = min(transformed_box[:, :, 1])[0]

        if min_col < 0:
            transformed_box[:, :, 0] -= min_col
            M[0, 2] -= min_col

        if min_row < 0:
            transformed_box[:, :, 1] -= min_row
            M[1, 2] -= min_row

        max_col = max(transformed_box[:, :, 0])[0]
        max_row = max(transformed_box[:, :, 1])[0]

        new_transformed_img = cv2.warpAffine(
            img, M, (int(max_col), int(max_row)))
        cv2.imshow('new affine', new_transformed_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
