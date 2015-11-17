'''
Many are borrowed from http://www.jhlabs.com/ip/filters/index.html
'''
from __future__ import print_function, division
import cv2
import numpy
import pdb

def horizontal_flip_map(rows, cols):
    map_x = numpy.fromfunction(lambda r, c: cols - 1 - c, (rows, cols), dtype=numpy.float32)
    map_y = numpy.fromfunction(lambda r, c: r, (rows, cols), dtype=numpy.float32)

    return map_x, map_y

def vertical_flip_map(rows, cols):
    map_x = numpy.fromfunction(lambda r, c: c, (rows, cols), dtype=numpy.float32)
    map_y = numpy.fromfunction(lambda r, c: rows - 1 - r, (rows, cols), dtype=numpy.float32)

    return map_x, map_y

def _water_map_function(r, c, c_r, c_c, squared_radius, amplitude, wavelenth, phase):
    dx = c - c_c
    dy = r - c_r

    squared_distance = dx ** 2 + dy ** 2
    if squared_distance > squared_radius:
        return (c, r)
    else:
        distance = numpy.sqrt(squared_distance)
        radius = numpy.sqrt(squared_radius)
        amount = amplitude * numpy.sin(distance / wavelenth * numpy.pi * 2 - phase)
        amount *= (radius - distance) / radius

        if distance != 0:
            amount *= wavelenth / distance

        return (c + dx * amount, r + dy * amount)

def water_ripple_map(rows, cols, amplitude, wavelenth, phase):
    radius = min(rows, cols) // 2
    squared_radius = radius ** 2
    c_r, c_c = rows // 2, cols // 2

    water_map = numpy.frompyfunc(lambda r, c: _water_map_function(r, c, c_r, c_c, squared_radius, amplitude, wavelenth, phase), 2, 2).outer(numpy.arange(rows), numpy.arange(cols))
    return water_map[0].astype(numpy.float32), water_map[1].astype(numpy.float32)

def _twist_map_function(r, c, c_r, c_c, squared_radius, angle):
    dx = c - c_c
    dy = r - c_r

    squared_distance = dx ** 2 + dy ** 2
    if squared_distance > squared_radius:
        return (c, r)
    else:
        distance = numpy.sqrt(squared_distance)
        radius = numpy.sqrt(squared_radius)
        a = numpy.arctan2(dy, dx) + angle * (radius - distance) / radius

        return (c_c + distance * numpy.cos(a), c_r + distance * numpy.sin(a))

def twist_ripple_map(rows, cols, angle):
    radius = min(rows, cols) // 2
    squared_radius = radius ** 2
    c_r, c_c = rows // 2, cols // 2

    water_map = numpy.frompyfunc(lambda r, c: _twist_map_function(r, c, c_r, c_c, squared_radius, angle), 2, 2).outer(numpy.arange(rows), numpy.arange(cols))
    return water_map[0].astype(numpy.float32), water_map[1].astype(numpy.float32)


img = cv2.imread('./data/lenna.png')
rows, cols = img.shape[:2]

map_x, map_y = horizontal_flip_map(rows, cols)

while True:
    k = cv2.waitKey(1) & 0xFF

    if k == ord('1'):
        map_x, map_y = horizontal_flip_map(rows, cols)
    elif k == ord('2'):
        map_x, map_y = vertical_flip_map(rows, cols)
    elif k == ord('3'):
        map_x, map_y = water_ripple_map(rows, cols, 10, 30, numpy.pi)
    elif k == ord('4'):
        map_x, map_y = twist_ripple_map(rows, cols, numpy.pi)
    elif k == 27:
        break

    remapped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)
    cv2.imshow('remapped', remapped)
