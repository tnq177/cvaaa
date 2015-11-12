"""Panography

This is a simple implementation of the paper Scene Collage & Flexible camera arrays
http://www1.cs.columbia.edu/CAVE/publications/pdfs/Nomura_EUROGRAPHICS07.pdf

It takes a sequence of overlapping images and returns a panograph built from
these images. Irrelevant images which do not have enough overlap with other images
are discarded.

Since this is just a simple implementation, it consists only these steps:

1/ Extract features from each image
2/ Match features between each pair of images
3/ Use RANSAC to extract only inliers of these features which satisfy the similarity
transform model
4/ Use these inliers to minimize the SSD (function 2 in the paper) to find the
set of transform matrix for each image
5/ Apply the transformations & merged the transformed images into a big one

Data taken from http://www1.cs.columbia.edu/CAVE/projects/scene_collage/imagegallery.php

Update:
So there are quite a handful of parameters to configure. I admit defeat.
Just some config for the test data:

panograph:
    ~50 seconds
    Commentary: Hmm, because these images contain some part of cloud texture, I think increasing 
    contrastThreshold and using more features from overlapping areas(increase size) could help?
    resize_height=640, bnnRatio=0.8, contrastThreshold=0.1, ransacThreshold=10.0, min_samples=8, max_trials=1000

panograph_2:
    ~200 seconds
    resize_height=360, bnnRatio=0.8, contrastThreshold=0.001, ransacThreshold=10.0, min_samples=4, max_trials=1000

panograph_3:
    ~8 seconds
    resize_height=360, bnnRatio=0.9, contrastThreshold=0.1, ransacThreshold=10.0, min_samples=4, max_trials=1000

panograph_4: 
    ~44 seconds
    resize_height=240, bnnRatio=0.8, contrastThreshold=0.04, ransacThreshold=11.0, min_samples=4, max_trials=1000

panograph_5: 
    ~37 seconds
    resize_height=640, bnnRatio=0.7, contrastThreshold=0.1, ransacThreshold=11.0, min_samples=4, max_trials=1000

panograph_6: 
    ~121 seconds
    Geez a different config takes only 60 seconds T.T, but I forgot it -.-"
    resize_height=360, bnnRatio=0.8, contrastThreshold=0.01, ransacThreshold=10.0, min_samples=4, max_trials=1000

panograph_7: 
    ~132 seconds
    resize_height=360, bnnRatio=0.8, contrastThreshold=0.01, ransacThreshold=10.0, min_samples=4, max_trials=1000)

panograph_8:
    ~20 seconds
    Commentary: These images seem to have low contrast, it seems like lowering the contrastThreshold helps with 
    finding features & fitting 
    resize_height=240, bnnRatio=0.8, contrastThreshold=0.01, ransacThreshold=11.0, min_samples=8, max_trials=1000

panograph_9:
    ~98 seconds
    resize_height=240, bnnRatio=0.8, contrastThreshold=0.04, ransacThreshold=11.0, min_samples=4, max_trials=1000
"""
from __future__ import print_function, division
import pdb
import datetime
import numpy
import cv2
from os import listdir
from os.path import isfile, join
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
from lmfit import minimize, Parameters
import timeit
from random import randint


class Image(object):

    def __init__(self, image_path, index, resize_height=320, contrastThreshold=0.04):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError('Image not found')
        self.image_path = image_path
        scale_factor = round(resize_height * 10 / min(self.img.shape[:2])) / 10
        self.img = cv2.resize(self.img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.index = index
        self.M = None

        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrastThreshold)
        self.kp, self.des = sift.detectAndCompute(self.gray, None)


class Panography(object):

    def __init__(self, images_directory_path, resize_height=320, bnnRatio=0.8, contrastThreshold=0.04, ransacThreshold=10.0, min_samples=6, max_trials=1000):
        self.bnnRatio = bnnRatio
        self.contrastThreshold = contrastThreshold
        self.ransacThreshold = ransacThreshold
        self.min_samples = min_samples
        self.max_trials = max_trials
        self.resize_height = resize_height

        image_paths = [join(images_directory_path, f) for f in listdir(images_directory_path) if isfile(
            join(images_directory_path, f)) and f.endswith(('.png', '.jpg', '.JPG'))]
        self.images = [Image(image_path, index, resize_height=self.resize_height, contrastThreshold=self.contrastThreshold)
                       for index, image_path in enumerate(image_paths)]

    def _get_largest_blob(self):
        '''
        Meant to find the largest set of images that are connected
        Not implemented (previous implementation was wrong)
        '''
        self.connected_indices = numpy.array(
            numpy.where(self.connected == 255))
        self.unique_indices = list(
            set(numpy.array(self.connected_indices).flatten()))
        print(self.unique_indices)

    def _extract_feature_pairs(self):
        length = len(self.images)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.connected = numpy.zeros((length, length), dtype=numpy.uint8)

        self.feature_points = {}
        for i in xrange(length - 1):
            self.feature_points[i] = {}
            for j in xrange(i + 1, length):
                self.feature_points[i][j] = [[], []]

        for i in xrange(length - 1):
            for j in xrange(i + 1, length):
                image_1 = self.images[i]
                image_2 = self.images[j]

                matches = bf.knnMatch(image_1.des, image_2.des, k=2)
                good = [m for m, n in matches if m.distance < self.bnnRatio * n.distance]

                # Not enough good points
                if len(good) < 10:
                    print("{0} NOT ENOUGH {1}".format(
                        image_1.image_path, image_2.image_path))
                    continue

                src_pts = numpy.float32(
                    [image_1.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = numpy.float32(
                    [image_2.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                if cv2.estimateRigidTransform(src_pts, dst_pts, False) is None:
                    print("{0} FAIL {1}".format(
                        image_1.image_path, image_2.image_path))
                    continue

                src_pts = src_pts.reshape(-1, 2)
                dst_pts = dst_pts.reshape(-1, 2)
                # Now use skimage ransac to get the inliers
                model_robust, inliers = ransac((src_pts, dst_pts), SimilarityTransform, min_samples=self.min_samples, residual_threshold=self.ransacThreshold, max_trials=self.max_trials)
                if len(inliers[inliers]) < 8:
                    continue

                # print(len(inliers[inliers]))
                # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                #    singlePointColor = None,
                #    matchesMask = inliers.ravel().tolist(), # draw only inliers
                #    flags = 2)

                # img3 = cv2.drawMatches(image_1.img, image_1.kp, image_2.img, image_2.kp, good,None,**draw_params)
                # cv2.imshow('inliers', img3)
                # cv2.waitKey()

                print("{0} --> {1}: {2} Inliers, RATIO {3}".format(i, j, len(src_pts[inliers]), len(src_pts[inliers]) / len(src_pts)))
                src_pts = src_pts[inliers]
                dst_pts = dst_pts[inliers]
                one_column = numpy.ones(
                    (src_pts.shape[0], 1), dtype=numpy.float32)
                src_pts = numpy.hstack((src_pts, one_column))
                dst_pts = numpy.hstack((dst_pts, one_column))

                self.feature_points[image_1.index][
                    image_2.index] = [src_pts, dst_pts]
                self.connected[image_1.index][image_2.index] = 255

    def _residuals(self, params):
        res = []
        for i, j in zip(self.connected_indices[0], self.connected_indices[1]):
            a1 = params["a{0}".format(i)].value
            b1 = params["b{0}".format(i)].value
            tx1 = params["tx{0}".format(i)].value
            ty1 = params["ty{0}".format(i)].value

            a2 = params["a{0}".format(j)].value
            b2 = params["b{0}".format(j)].value
            tx2 = params["tx{0}".format(j)].value
            ty2 = params["ty{0}".format(j)].value
            src_M = numpy.float32([[a1, -b1], [b1, a1], [tx1, ty1]])
            dst_M = numpy.float32([[a2, -b2], [b2, a2], [tx2, ty2]])

            temp = self.feature_points[i][j][0].dot(src_M) - self.feature_points[i][j][1].dot(dst_M)
            temp = temp ** 2
            res.extend(numpy.sqrt(temp[:, 0] + temp[:, 1]))

        # Print out to see the optimization process
        # print(sum(res))
        return res

    def _calculate_layout(self):
        params = Parameters()

        pivot_idx = self.unique_indices[0]
        params.add("a{0}".format(pivot_idx), value=1.0, vary=False)
        params.add("b{0}".format(pivot_idx), value=0.0, vary=False)
        params.add("tx{0}".format(pivot_idx), value=0.0, vary=False)
        params.add("ty{0}".format(pivot_idx), value=0.0, vary=False)

        for i in xrange(1, len(self.unique_indices)):
            idx = self.unique_indices[i]
            params.add("a{0}".format(idx), value=1.0)
            params.add("b{0}".format(idx), value=0.0)
            params.add("tx{0}".format(idx), value=randint(0, 100))
            params.add("ty{0}".format(idx), value=randint(0, 100))

        print('---minimizing---')
        out = minimize(self._residuals, params)

        for image in self.images:
            i = image.index
            if i in self.unique_indices:
                a = out.params["a{0}".format(i)].value
                b = out.params["b{0}".format(i)].value
                tx = out.params["tx{0}".format(i)].value
                ty = out.params["ty{0}".format(i)].value
                image.M = numpy.float32([[a, b, tx], [-b, a, ty]])

    def get_layout(self):
        self._extract_feature_pairs()
        self._get_largest_blob()
        self._calculate_layout()

    def merge(self):
        min_row, min_col = 0, 0
        for image in self.images:
            if image.index in self.unique_indices:
                print('---finding final panograph size---')
                rows, cols = image.img.shape[:2]
                box = numpy.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1],
                                   [0, rows - 1]], dtype=numpy.float32).reshape(-1, 1, 2)
                transformed_box = cv2.transform(box, image.M)
                _min_col = min(transformed_box[:, :, 0])[0]
                _min_row = min(transformed_box[:, :, 1])[0]

                if _min_row < min_row:
                    min_row = _min_row
                if _min_col < min_col:
                    min_col = _min_col

        if min_row < 0:
            min_row = -min_row
        if min_col < 0:
            min_col = -min_col

        max_row, max_col = 0, 0
        for image in self.images:
            if image.index in self.unique_indices:
                print('---merging---')
                image.M[0, 2] += min_col
                image.M[1, 2] += min_row

                transformed_box = cv2.transform(box, image.M)
                _max_col = max(transformed_box[:, :, 0])[0]
                _max_row = max(transformed_box[:, :, 1])[0]

                if _max_row > max_row:
                    max_row = _max_row
                if _max_col > max_col:
                    max_col = _max_col

        result = numpy.zeros((max_row, max_col, 3), numpy.uint8)
        result.fill(255)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        for image in self.images:
            if image.index in self.unique_indices:
                transformed_img = cv2.warpAffine(image.img, image.M, (max_col, max_row), borderMode=cv2.BORDER_TRANSPARENT)
                transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2BGRA)
                numpy.copyto(result, transformed_img, where=numpy.logical_and(result == 255, transformed_img != 255))

        result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        return result

if __name__ == '__main__':
    s1 = timeit.default_timer()
    pano = Panography('./data/panograph_8', resize_height=360, bnnRatio=0.8, contrastThreshold=0.1, ransacThreshold=10.0, min_samples=4, max_trials=1000)
    pano.get_layout()
    result = pano.merge()
    s2 = timeit.default_timer()
    print('It takes {0} seconds'.format(s2 - s1))
    cv2.imwrite("{0}.jpg".format(datetime.datetime.now().strftime("%I:%M%p%s on %B %d, %Y")), result)
