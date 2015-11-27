from __future__ import print_function, division
import numpy
import cv2
from os import listdir
from os.path import isfile, join
from skimage.transform._geometric import GeometricTransform


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


def focal_from_homography(M):
    """Summary: From https://github.com/Itseez/opencv/blob/master/modules/stitching/src/autocalib.cpp#L66

    Args:
        M (3x3 ndarray): Homography matrix 

    Returns:
        Tuple: (fx, fy)
    """
    h = M.flatten()

    fy = None
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2

    if v1 < v2:
        v1, v2 = v2, v1

    if v1 > 0 and v2 > 0:
        fy = numpy.sqrt(v1 if numpy.abs(d1) > numpy.abs(d2) else v2)
    elif v1 > 0:
        fy = numpy.sqrt(v1)
    else:
        fy = None

    fx = None
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
    v1 = -h[2] * h[5] / d1
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2

    if v1 < v2:
        v1, v2 = v2, v1

    if v1 > 0 and v2 > 0:
        fx = numpy.sqrt(v1 if numpy.abs(d1) > numpy.abs(d2) else v2)
    elif v1 > 0:
        fx = numpy.sqrt(v1)
    else:
        fx = None

    return (fx, fy)


def estimate_focal_lengths(images, knnRatio=0.9, ransacThreshold=10):
    print('Estimating focal length')
    focal_lengths = []
    append = focal_lengths.append

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    for i in xrange(1, len(images)):
        image_1 = cv2.cvtColor(images[i - 1], cv2.COLOR_BGR2GRAY)
        image_2 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image_1, None)
        kp2, des2 = sift.detectAndCompute(image_2, None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < knnRatio * n.distance]

        if len(good) > 10:
            src_pts = numpy.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = numpy.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, ransacThreshold)
            if M is not None:
                fx, fy = focal_from_homography(M)
                if fx and fy:
                    append(numpy.sqrt(fx * fy))
                    print(numpy.sqrt(fx * fy))

    if focal_lengths:
        print('Got it: {0}'.format(numpy.median(focal_lengths)))
        return numpy.median(focal_lengths)
    else:
        print('Fail')
        return None


class TranslationTransform(GeometricTransform):

    def __init__(self):
        self.params = numpy.float32([0, 0])

    def __call__(self, src):
        return src + self.params

    def estimate(self, src, dst):
        diff = dst - src
        rows = src.shape[0]

        self.params = numpy.sum(diff, axis=0) / rows

        return True

    def residuals(self, src, dst):
        return numpy.sqrt(numpy.sum((self(src) - dst) ** 2, axis=1))


def get_laplacian_stack(img, levels=6):
    lap_stack = []
    temp_1 = img.copy()
    temp_2 = cv2.pyrDown(temp_1)

    for i in xrange(levels - 1):
        temp_2_up = cv2.pyrUp(temp_2)
        lap_stack.append(temp_1 - temp_2_up)

        temp_1 = temp_2
        temp_2 = cv2.pyrDown(temp_2)

    lap_stack.append(temp_1)

    return lap_stack[::-1]


def get_gaussian_stack(img, levels=6):
    gaus_stack = [img]

    temp = img.copy()
    for i in xrange(levels - 1):
        temp = cv2.pyrDown(temp)
        gaus_stack.append(temp)

    return gaus_stack[::-1]


def pyr_blending(stack_1, stack_2, mask_stack):
    '''
    Expect all images are of type numpy.float32
    '''
    stack = []
    for img1, img2, mask in zip(stack_1, stack_2, mask_stack):
        merged_img = img1 * mask + img2 * (1 - mask)
        stack.append(merged_img)

    length = len(stack)
    blended = stack[0]
    for i in range(1, length):
        blended = cv2.pyrUp(blended)
        blended += stack[i]

    return numpy.uint8(blended)
