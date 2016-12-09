# -*- coding: utf8 -*-

"""
detector.py
author: Septimus Liu
Provide QRCode detection, text region detection and face detection.
"""

import cv2
import math
import numpy as np
import os

TEXT_LAPLACIAN_THRESHOLD = 2000
TEXT_MAX_REGION = 3
TEXT_AREA_THRESHOLD = 0.09


def text_detect(img):
    """
    Detect if picture contains so many text regions
    :param img: image object
    :return: True - Too many text regions; False - Otherwise
    """
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pre-processing to bring out text
    dilation = _preproc_txt(gray)

    # Calculate area and number of text region
    region, area_text = _find_txt_region(dilation, gray)

    area_total = img.shape[0] * img.shape[1]
    area_portion = float(area_text) / float(area_total)

    return area_portion > TEXT_AREA_THRESHOLD or len(region) > TEXT_MAX_REGION


def qrcode_detect(img):
    """
    Detect if picture contains QR code
    :param img: image object
    :return: True - QR code has been detected; False - Otherwise
    """
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gb = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gb, 100, 200)
    th, bi_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    def cv_distance(P, Q):
        return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]),2)))

    def is_timing_pattern(line):
        # Remove white pixel on the begin and the end
        while line[0] != 0:
            line = line[1:]
        while line[-1] != 0:
            line = line[:-1]
        # Count continuing white or black pixels
        c = []
        count = 1
        l = line[0]
        for p in line[1:]:
            if p == l:
                count += 1
            else:
                c.append(count)
                count = 1
            l = p
        c.append(count)
        # If black and white interval is few
        if len(c) < 5:
            return False
        # Calculate variance to judge if it is a Timing Pattern
        threshold = 4
        return np.var(c) < threshold

    def check(a, b):
        # Store two endpoints of the shortest line a and b
        s1_ab = ()
        s2_ab = ()
        # Store distance of two endpoints of the shortest line a and b
        s1 = np.iinfo('i').max
        s2 = s1
        for ai in a:
            for bi in b:
                d = cv_distance(ai, bi)
                if d < s2:
                    if d < s1:
                        s1_ab, s2_ab = (ai, bi), s1_ab
                        s1, s2 = d, s1
                    else:
                        s2_ab = (ai, bi)
                        s2 = d
        a1, a2 = s1_ab[0], s2_ab[0]
        b1, b2 = s1_ab[1], s2_ab[1]

        a1 = (a1[0] + (a2[0]-a1[0])*1/14, a1[1] + (a2[1]-a1[1])*1/14)
        b1 = (b1[0] + (b2[0]-b1[0])*1/14, b1[1] + (b2[1]-b1[1])*1/14)
        a2 = (a2[0] + (a1[0]-a2[0])*1/14, a2[1] + (a1[1]-a2[1])*1/14)
        b2 = (b2[0] + (b1[0]-b2[0])*1/14, b2[1] + (b1[1]-b2[1])*1/14)

        # Pick out the shortest two lines
        try:
            line1 = [x[2] for x in _create_line_iterator(a1, b1, bi_img)]
            if is_timing_pattern(line1):
                return True
            else:
                line2 = [x[2] for x in _create_line_iterator(a2, b2, bi_img)]
                if is_timing_pattern(line2):
                    return True
        except Exception, err:
            pass
        return False

    # Pick out all QRcode-like contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or contours is None or not (len(hierarchy) and len(contours)):
        return False
    hierarchy = hierarchy[0]
    found = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c += 1
        # QR code may has contours nested more than 5 layers
        if c >= 5:
            found.append(i)

    boxes = []
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = np.asarray(cv2.cv.BoxPoints(rect))
        box = map(tuple, box.astype(int))
        boxes.append(box)

    # Store valid QR code endpoints
    valid = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if check(boxes[i], boxes[j]):
                valid.add(i)
                valid.add(j)

    if len(valid) > 0 and len(valid) % 3 == 0:
        return True
    else:
        return False


def _preproc_txt(gray):
    """
    Bring out text region, make them prepared to be find
    :param gray: gray-scaled image object
    :return: Dilated and erosed image object
    """
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    ele1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    ele2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # Dilate and erose contours to outline the text region
    dilation = cv2.dilate(binary, ele2, iterations=1)
    # Remove details like table lines
    erosion = cv2.erode(dilation, ele1, iterations=1)
    dilation2 = cv2.dilate(erosion, ele2, iterations=1)

    return dilation2


def _find_txt_region(img, gray):
    """
    Find text regions and calculate their area and numbers
    :param img: image object
    :param gray: gray-scaled image object
    :return: Region box points and their total area size
    """
    area_text = 0
    region = []
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # Exclude contours that are too small
        if area < 1000:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.asarray(box)

        # If the region is blurry, it may not be text region
        lap = cv2.Laplacian(gray[box[1][1]:box[0][1], box[0][0]:box[3][0]], cv2.CV_64F)
        if lap is None or lap.var() < TEXT_LAPLACIAN_THRESHOLD:
            continue

        box = box.astype(int)
        x0 = box[0][0] if box[0][0] > 0 else 0
        x1 = box[2][0] if box[2][0] > 0 else 0
        y0 = box[0][1] if box[0][1] > 0 else 0
        y1 = box[2][1] if box[2][1] > 0 else 0
        height = abs(y0 - y1)
        width = abs(x0 - x1)

        # Exclude vertical rectangle
        if height > width * 0.3:
            continue
        area_text += height * width
        region.append(box)

    return region, area_text


def _create_line_iterator(p1, p2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """  # define local variables for readability
    image_h = img.shape[0]
    image_w = img.shape[1]
    p1_x = p1[0]
    p1_y = p1[1]
    p2_x = p2[0]
    p2_y = p2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    d_x = p2_x - p1_x
    d_y = p2_y - p1_y
    d_x_a = np.abs(d_x)
    d_y_a = np.abs(d_y)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(d_y_a, d_x_a), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    neg_y = p1_y > p2_y
    neg_x = p1_x > p2_x
    if p1_x == p2_x:  # vertical line segment
        itbuffer[:, 0] = p1_x
        if neg_y:
            itbuffer[:, 1] = np.arange(p1_y - 1, p1_y - d_y_a - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(p1_y + 1, p1_y + d_y_a + 1)
    elif p1_y == p2_y:  # horizontal line segment
        itbuffer[:, 1] = p1_y
        if neg_x:
            itbuffer[:, 0] = np.arange(p1_x - 1, p1_x - d_x_a - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(p1_x + 1, p1_x + d_x_a + 1)
    else:  # diagonal line segment
        steep_slope = d_y_a > d_x_a
        if steep_slope:
            slope = d_x.astype(np.float32) / d_y.astype(np.float32)
            if neg_y:
                itbuffer[:, 1] = np.arange(p1_y - 1, p1_y - d_y_a - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(p1_y + 1, p1_y + d_y_a + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - p1_y)).astype(np.int) + p1_x
        else:
            slope = d_y.astype(np.float32) / d_x.astype(np.float32)
            if neg_x:
                itbuffer[:, 0] = np.arange(p1_x - 1, p1_x - d_x_a - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(p1_x + 1, p1_x + d_x_a + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - p1_x)).astype(np.int) + p1_y

            # Remove points outside of image
    col_x = itbuffer[:, 0]
    col_y = itbuffer[:, 1]
    itbuffer = itbuffer[(col_x >= 0) & (col_y >= 0) & (col_x < image_w) & (col_y < image_h)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer
