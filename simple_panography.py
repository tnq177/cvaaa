from __future__ import print_function, division
import numpy
import cv2
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
from scipy.optimize import leastsq

img1 = cv2.imread('./data/yosemite1.jpg')
img2 = cv2.imread('./data/yosemite2.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des2, des1, k=2)
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = numpy.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 2)
dst_pts = numpy.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)

model_robust, inliers = ransac((src_pts, dst_pts), SimilarityTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
M1 = model_robust.params[:2, :]

# Try to use least square method to find M instead
src_pts = src_pts[inliers]
dst_pts = dst_pts[inliers]

def residuals(params):
    global src_pts, dst_pts
    a, b, tx, ty = params

    M = numpy.float32([[a, b, tx], [-b, a, ty]])
    one_column = numpy.ones((src_pts.shape[0], 1), dtype=numpy.float32)
    _src_pts = numpy.hstack((src_pts, one_column))

    res = _src_pts.dot(M.T) - dst_pts
    res = res ** 2
    res = numpy.sqrt(res[:, 0] + res[:, 1])
    return res

params = [1.0, 0.0, 100.0, 100.0]
out = leastsq(residuals, params)[0]
M2 = numpy.float32([[out[0], out[1], out[2]], [-out[1], out[0], out[3]]])


def getPano(M, img1, img2):
    rows, cols = img2.shape[:2]
    # This is the tricky part. For transform, the point is identified as pair
    # of (col, row) instead of (row, col)
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

    I = numpy.array([[1, 0, 0], [0, 1, 0]], dtype=numpy.float)
    transformed_img1 = cv2.warpAffine(img1, I, (max_col, max_row))
    transformed_img2 = cv2.warpAffine(img2, M, (max_col, max_row))
    numpy.copyto(
        transformed_img1, transformed_img2, where=transformed_img1 == 0)

    return transformed_img1

M1_pano = getPano(M1, img1, img2)
M2_pano = getPano(M2, img1, img2)

cv2.imshow('With Skimage & OpenCV', M1_pano)
cv2.imshow('With M found by leastsq', M2_pano)
cv2.imshow('diff', cv2.subtract(M1_pano, M2_pano))
cv2.waitKey()
