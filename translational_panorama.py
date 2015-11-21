"""
Simple program to create panorama

1/ Remap to spherical
2/ Use ransac to find translation 
3/ Calculate final layout and composite

Test data taken from various sources. I do not own them.
"""
from __future__ import print_function, division
import sys
import time
import pdb
import numpy
import cv2
from os import path
from natsort import natsorted
from skimage.measure import ransac
from common_utils import focal_from_homography, get_images_paths, estimate_focal_lengths, TranslationTransform


class Image(object):

    def __init__(self, img, contrastThreshold=0.04):
        self.img = img
        if self.img is None:
            raise ValueError("Yo, give me a valid image")

        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrastThreshold)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.kp, self.des = sift.detectAndCompute(gray, None)
        self.M = None


class TranslationalPanorama(object):

    def __init__(self, images_paths, f, k1, k2, knnRatio=0.8, ransacThreshold=10):
        print('Init')
        self.images_paths = images_paths
        self.f = f
        self.k1 = k1
        self.k2 = k2
        self.knnRatio = knnRatio
        self.ransacThreshold = ransacThreshold

    def _build_spherical_map(self):
        """Summary: Build spherical map_x, map_y to warp images into spherical
        coordinates. Really expect the images to have the same shape.

        Returns:
            tuple of ndarrays: (map_x, map_y)
        """
        print('Building spherical map from original to spherical coordinates using f, k1, k2')
        h, w = cv2.imread(self.images_paths[0], 0).shape
        # Calculate the final shape of warped image
        w_ = 2 * self.f * numpy.arctan2(w, (2 * self.f))
        h_ = 2 * self.f * numpy.arctan2(h, (2 * self.f))

        # # Roughly account for the lens distortion correction
        # rc2 = (w_ / (2 * self.f)) ** 2 + (h_ / (2 * self.f)) ** 2
        # w_ = int(w_ * (1 + self.k1 * rc2 + self.k2 * rc2 ** 2))
        # h_ = int(h_ * (1 + self.k1 * rc2 + self.k2 * rc2 ** 2))

        w_, h_ = int(w_), int(h_)
        self.map_x = numpy.zeros((h_, w_), dtype=numpy.float32)
        self.map_y = numpy.zeros((h_, w_), dtype=numpy.float32)
        for x in xrange(w_):
            for y in xrange(h_):
                xc = (x - w_/2) / self.f
                yc = (y - h_/2) / self.f

                if self.k1 is not None and self.k2 is not None:
                    rc2 = xc ** 2 + yc ** 2
                    xc = xc * (1 + self.k1 * rc2 + self.k2 * rc2 ** 2)
                    yc = yc * (1 + self.k1 * rc2 + self.k2 * rc2 ** 2)

                p0 = numpy.tan(xc)
                p1 = numpy.tan(yc)

                self.map_x[y, x] = w/2 + self.f * p0
                self.map_y[y, x] = h/2 + p1 * self.f * numpy.sqrt(1 + p0 ** 2)

    def warp_spherical(self):
        self._build_spherical_map()
        print('Warping to spherical coordinates')
        self.images = []
        for image_path in self.images_paths:
            img = cv2.imread(image_path)
            spherical = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            self.images.append(Image(spherical))

    def calc_transformations(self):
        print('Calculating each pair translation matrix')
        self.images[0].M = numpy.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        for i in xrange(1, len(self.images)):
            image_1 = self.images[i]
            image_2 = self.images[i - 1]

            matches = bf.knnMatch(image_1.des, image_2.des, k=2)
            good = [m for m, n in matches if m.distance <
                    self.knnRatio * n.distance]

            src_pts = numpy.float32(
                [image_1.kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
            dst_pts = numpy.float32(
                [image_2.kp[m.trainIdx].pt for m in good]).reshape(-1, 2)

            model_robust, _ = ransac((src_pts, dst_pts), TranslationTransform,
                                     min_samples=6, residual_threshold=self.ransacThreshold, max_trials=1000)
            tx, ty = model_robust.params
            M = numpy.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            image_1.M = M.dot(image_2.M)
            print(image_1.M)

    def merge(self):
        min_row, min_col = 0, 0
        for i in [0, len(self.images) - 1]:
            image = self.images[i]
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
        for i in [0, len(self.images) - 1]:
            image = self.images[i]
            print('---finding final panograph size---')
            M = image.M.copy()
            M[0, 2] += min_col
            M[1, 2] += min_row

            transformed_box = cv2.transform(box, M)
            _max_col = max(transformed_box[:, :, 0])[0]
            _max_row = max(transformed_box[:, :, 1])[0]

            if _max_row > max_row:
                max_row = _max_row
            if _max_col > max_col:
                max_col = _max_col

        result = numpy.zeros((max_row, max_col, 3), numpy.uint8)
        result.fill(255)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        print('Merging')
        for image in self.images:
            image.M[0, 2] += min_col
            image.M[1, 2] += min_row
            transformed_img = cv2.warpPerspective(
                image.img, image.M, (max_col, max_row), borderMode=cv2.BORDER_TRANSPARENT)
            transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2BGRA)
            numpy.copyto(result, transformed_img, where=numpy.logical_and(
                result == 255, transformed_img != 255))

        result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        return result

if __name__ == '__main__':
    for name in ['carmel', 'diamondhead', 'fishbowl', 'goldengate', 'halfdome', 'hotel', 'office', 'rio', 'shanghai', 'yard']:
        images_dir = './data/Panorama/adobe_panoramas/data/' + name
        images_paths = natsorted(get_images_paths(images_dir))

        # Read/Estimate focal length
        f = k1 = k2 = None
        if path.exists(path.join(images_dir, 'info.txt')):
            with open(path.join(images_dir, 'info.txt')) as f:
                for line in f:
                    key, val = line.split()
                    if key == 'f':
                        f = float(val)
                    elif key == 'k1':
                        k1 = float(val)
                    elif key == 'k2':
                        k2 = float(val)
        else:
            images = [cv2.imread(image_path) for image_path in images_paths]
            f = estimate_focal_lengths(images, knnRatio=0.8, ransacThreshold=10)

        if f is None:
            print("Cannot do anything without a focal length")
        else:
            panorama = TranslationalPanorama(images_paths, f, k1, k2)
            panorama.warp_spherical()
            panorama.calc_transformations()
            result = panorama.merge()
            cv2.imwrite(path.join('./data/Panorama/results', path.basename(images_dir) + '.png'), result)
            cv2.waitKey()
