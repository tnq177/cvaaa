'''
http://mathworld.wolfram.com/StereographicProjection.html
Test image: https://www.idolbin.com/gprofile/104524111022384695688
I do not own the image
'''
from __future__ import print_function, division
import numpy
import cv2
from common_utils import trackbar_changed_do_nothing


if __name__ == '__main__':
    img = cv2.imread('./data/photosphere.jpg')
    img = cv2.resize(
        img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LANCZOS4)
    little_planet = img.copy()
    h, w = img.shape[:2]

    base_map_x, base_map_y = numpy.float32(
        numpy.meshgrid(numpy.arange(w), numpy.arange(h)))

    cv2.namedWindow('little planet')
    cv2.createTrackbar(
        'longitude', 'little planet', 0, 360, trackbar_changed_do_nothing)
    cv2.createTrackbar(
        'latitude', 'little planet', 0, 180, trackbar_changed_do_nothing)
    cv2.createTrackbar(
        'z', 'little planet', 16, 100, trackbar_changed_do_nothing)

    previous_lon = -1
    previous_lat = -1
    previous_z = -1
    while True:
        lon = cv2.getTrackbarPos('longitude', 'little planet')
        lat = cv2.getTrackbarPos('latitude', 'little planet')
        z = cv2.getTrackbarPos('z', 'little planet')

        if lon != previous_lon or lat != previous_lat or z != previous_z:
            previous_lon, previous_lat, previous_z = lon, lat, z

            if z == 0:
                z = 1
            R = w / z
            phi0 = lat * numpy.pi/180 - numpy.pi/2
            lamda0 = lon * numpy.pi/180
            X = base_map_x - w/2
            Y = base_map_y - h/2
            r = numpy.sqrt(X ** 2 + Y ** 2)
            c = 2 * numpy.arctan(r / (2 * R))

            map_lat = numpy.pi/2 + numpy.arcsin(
                numpy.cos(c) * numpy.sin(phi0) + Y * numpy.sin(c) * numpy.cos(phi0) / r)
            map_lon = lamda0 + numpy.arctan2(X * numpy.sin(c), r * numpy.cos(
                phi0) * numpy.cos(c) - Y * numpy.sin(phi0) * numpy.sin(c))

            map_lon = numpy.mod(map_lon + numpy.pi, 2 * numpy.pi)

            map_y = (h - 1) * map_lat / numpy.pi
            map_x = (w - 1) * map_lon / (2 * numpy.pi)

            little_planet = cv2.remap(
                img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)

        cv2.imshow('little planet', little_planet)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
