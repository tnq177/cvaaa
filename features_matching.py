# Not my code, taken from OpenCV Tutorial
import numpy
import cv2


if __name__ == '__main__':
    img1 = cv2.imread('./data/iphone_1.JPG', 0)
    img2=cv2.imread('./data/iphone_2.JPG', 0)

    # ORB detector
    orb=cv2.ORB_create()

    # Key points & descriptor
    kp1, des1=orb.detectAndCompute(img1, None)
    kp2, des2=orb.detectAndCompute(img2, None)

    # BFMatcher with Hamming distance
    # Return the nearest neighbor only
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Matchers
    matches=bf.match(des1, des2)

    # Match the descriptors
    matches=sorted(matches, key = lambda x: x.distance)

    # Draw first 10 matches
    for i in xrange(10):
        img3 = numpy.zeros(img1.shape)
        img3 = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[i: i+1], img3, flags = 2)
        cv2.imshow('img', img3)
        cv2.waitKey()
