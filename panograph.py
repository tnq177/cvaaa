# This program assumes the feed images are overlapped
# enough to generate a panograph

from __future__ import print_function, division
import numpy
import cv2
from os import listdir
from os.path import isfile, join


class ImageHolder(object):

    def __init__(self, image_path, index, scale=0.25):
        self.image = cv2.imread(image_path)
        self.color = cv2.resize(
            self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        self.M = None
        self.merged = False
        self.index = index

        surf = cv2.xfeatures2d.SIFT_create()
        self.kp, self.des = surf.detectAndCompute(self.gray, None)


class Panography(object):

    def __init__(self, directory):
        self.directory = directory
        image_paths = [join(directory, f)
                       for f in listdir(directory) if isfile(join(directory, f)) and 'JPG' in f]
        self.images = [ImageHolder(image_path, index)
                       for index, image_path in enumerate(image_paths)]
        self.merged_images = []
        self.matchDict = {}

    def find_transform_matrix(self):
        pivot_image = self.images.pop()
        pivot_image.M = numpy.array([[1, 0, 0], [0, 1, 0]]).astype(numpy.float32)
        pivot_image.merged = True 
        self.merged_images.append(pivot_image)

        self.shape = list(pivot_image.gray.shape)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        while len(self.images) > 0:
            print("Images left to merge {0}".format(len(self.images)))
            print("Images merged so far {0}".format(len(self.merged_images)))
            image = self.images.pop()

            for merged_image in self.merged_images:
                matches = flann.knnMatch(image.des, merged_image.des, k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                if len(good) > 10:
                    src_pts = numpy.float32([ image.kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = numpy.float32([ merged_image.kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    min_size = min(len(src_pts), len(dst_pts))
                    affine_M = cv2.estimateRigidTransform(src_pts[:min_size], dst_pts[:min_size], fullAffine=False)

                    if affine_M is None:
                        break

                    h, w = image.gray.shape 
                    image_box = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    affined_image_box = cv2.transform(image_box, affine_M)
                    min_r = min(affined_image_box[:, :, 0])[0]
                    min_c = min(affined_image_box[:, :, 1])[0]

                    if min_r < 0:
                        min_r = -min_r
                        self.shape[0] = self.shape[0] + min_r
                    else:
                        min_r = 0

                    if min_c < 0:
                        min_c = -min_c
                        self.shape[1] = self.shape[1] + min_c
                    else:
                        min_c = 0

                    compensate_M = numpy.array([[1, 0, min_c], [0, 1, min_r], [0, 0, 1]], dtype=numpy.float32)
                    image.M = affine_M.dot(compensate_M)
                    image.merged = True 
                    self.merged_images.append(image)
                    # cv2.imshow('image', image.color)
                    # cv2.imshow('affined', cv2.warpAffine(image.color, image.M, dsize=(int(self.shape[1]), int(self.shape[0]))))
                    # cv2.waitKey()
                    break
            else:   
                self.images.append(image)

    def merge(self):
        self.shape = [int(i) for i in self.shape]
        result = numpy.zeros((self.shape[0], self.shape[1], 3), dtype=numpy.float32)
        result.fill(-1)

        for merged_image in self.merged_images:
            transform_image = cv2.warpAffine(merged_image.color, merged_image.M, dsize=(int(self.shape[1]), int(self.shape[0])))
            numpy.copyto(result, transform_image, where=((result == -1) & (transform_image !=0)))
            cv2.imshow('transform_image', transform_image)
            # temp_result = numpy.zeros(result.shape, dtype=numpy.float32)
            # numpy.copyto(temp_result, transform_image, where=(result == -1))
            # result = result + temp_result + 1
            cv2.imshow('result', numpy.uint8(result))
            cv2.waitKey()

        result[result == -1] = 0 
        result = result.astype(numpy.uint8)
        cv2.imshow('result', result)
        cv2.waitKey()

panograph = Panography('data/panograph')
panograph.find_transform_matrix()
panograph.merge()
