'''
Just for future reference.

System: OSX 10.10.5, Mac mini late 2012, quad core i7, 4GB RAM, Intel HD Graphics 4000 1024 MB

A few years ago getting Kinect to work on Mac was such a pain in the ass. Then came 
`totakke` https://github.com/totakke/homebrew-openni2. 
Just tapped & used his formula. I tested on this mac when it was still with 10.8 or 10.9,
I guess. It worked flawlessly. The stream was as smooth as expected on Windows.

Then after the formula was brought to homebrew-science, it never worked for me again.
Not sure why. I actually have to do the following:

1/ libusb installed from homebrew 
2a/ libfreenect & openni2 from homebrew-science never worked
2b/ Clone libfreenect from OpenKinect: https://github.com/OpenKinect/libfreenect
    * To build with OpenNI2-freenectdriver, turn on flag BUILD_OPENNI2_DRIVER
    * Copy the built driver to OpenNI folder (step 3)
3/ OpenNI2 download from http://structure.io/openni, Install using provided script, remember to add variables to path

With above steps, the Kinect works on Mac. I tried to use the primesense python wrapper as below.

Problems?: It runs for like a minute or two then crashes. No idea how to fix. 

Freenect provided executables run flawlessly, never crash.
'''
import numpy
import cv2
from primesense import openni2


def show_depth_value(event, x, y, flags, param):
    global depth
    print(depth[y, x])

if __name__ == '__main__':
    # can also accept the path of the OpenNI redistribution
    openni2.initialize()

    dev = openni2.Device.open_any()

    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    color_stream = dev.create_color_stream()
    color_stream.start()

    depth_scale_factor = 255.0 / depth_stream.get_max_pixel_value()

    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', show_depth_value)

    while True:
        # Get depth
        depth_frame = depth_stream.read_frame()
        h, w = depth_frame.height, depth_frame.width
        depth = numpy.ctypeslib.as_array(
            depth_frame.get_buffer_as_uint16()).reshape(h, w)
        depth_uint8 = cv2.convertScaleAbs(depth, alpha=depth_scale_factor)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_HSV)

        # Get color
        color_frame = color_stream.read_frame()
        color = numpy.ctypeslib.as_array(
            color_frame.get_buffer_as_uint8()).reshape(h, w, 3)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Display
        cv2.imshow('depth', depth_uint8)
        cv2.imshow('depth colored', depth_colored)
        cv2.imshow('color', color)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    depth_stream.stop()

    openni2.unload()
