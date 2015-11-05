from __future__ import print_function, division
import numpy
import cv2

img1 = cv2.imread('./data/yosemite1.jpg')
img2 = cv2.imread('./data/yosemite2.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des2, des1, k=2)
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts = numpy.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = numpy.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=False)

rows, cols = img2.shape[:2]
# This is the tricky part. For transform, the point is identified as pair of (col, row) instead of (row, col)
box = numpy.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]], dtype=numpy.float32).reshape(-1, 1, 2)
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
numpy.copyto(transformed_img1, transformed_img2, where=transformed_img1==0)

cv2.imshow('simple pano from 2 images', transformed_img1)
cv2.waitKey()
