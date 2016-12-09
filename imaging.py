# -*- coding: utf8 -*-

"""
imaging.py
author: Septimus Liu
Crop image automatically from either a remote address or base64 encoded stream.
"""

import argparse
import httplib2
import cStringIO
import cv2
import datetime
import math
import numpy as np
import os
from PIL import Image
import random
import sys
import time


_casc_path = 'conf/haarcascade_frontalface_default.xml'
_data_path = 'data'


def crop_image(url, width=0, height=0, img_type='jpg', quality=100, **kwargs):
    """
    :param url: source image url
    :param width: cropped width
    :param height: cropped height
    :param img_type: cropped image type
    :param quality: cropped image quality
    """
    face_detect = kwargs.get('face_detect')
    if not os.path.exists(_data_path):
        os.mkdir(_data_path)
    h = httplib2.Http('.cache')
    try:
        (resp, content) = h.request(url, 'GET')
        if content[0:20].find('The requested URL') != -1:
            raise httplib2.HttpLib2Error('URL cannot find in server')
    except httplib2.HttpLib2Error, err:
        raise Exception('%s request error: %s' % (url, str(err)))

    try:
        # Fetch image stream from response
        img = Image.open( cStringIO.StringIO(content))
    except Exception, err:
        raise Exception('%s opening image failed: %s' % (url, err))

    try:
        img_new = _get_sized_img(img, content, width, height, face_detect)
    except Exception, err:
        raise Exception('Sizing image_url: %s error: %s' % (url, err))

    # Generate a random prefix for image file
    img_prefix = _gen_prefix()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(_data_path, dir_prefix.strftime('%Y%m%d'))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    subdir_path = os.path.join(dir_path, dir_prefix.strftime('%H'))
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)

    local_path = os.path.join(subdir_path, img_localname)
    if os.path.exists(local_path):
        img_prefix = _gen_prefix()
        img_localname = '%s.%s' % (img_prefix, img_type)
        local_path = os.path.join(subdir_path, img_localname)
        if os.path.exists(local_path):
            raise Exception('%s image name existed' % local_path)

    try:
        # Save the image
        try:
            img_new.save(local_path, quality=quality)
        except IOError:
            img_new.convert('RGB').save(local_path, quality=quality)
    except Exception, err:
        raise Exception('%s saving image failed: %s' % (url, err))

    return local_path


def crop_image_file(filename, width=0, height=0, img_type='jpg', quality=100, **kwargs):
    """
    :param url: source image base64 encoded stream
    :param width: cropped width
    :param height: cropped height
    :param img_type: cropped image type
    :param quality: cropped image quality
    """
    face_detect = kwargs.get('face_detect')
    if not os.path.exists(_data_path):
        os.mkdir(_data_path)

    try:
        # Fetch image stream from response
        img = Image.open(filename)
    except Exception, err:
        raise Exception('%s opening image file failed: %s' % (filename, err))

    try:
        fp = open(filename, 'rb')
        content = fp.read()
        img_new = _get_sized_img(img, content, width, height, face_detect)
        fp.close()
    except Exception, err:
        raise Exception('Sizing image file: %s error: %s' % (filename, err))

    # Generate a random prefix for image file
    img_prefix = _gen_prefix()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(_data_path, dir_prefix.strftime('%Y%m%d'))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    subdir_path = os.path.join(dir_path, dir_prefix.strftime('%H'))
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)

    local_path = os.path.join(subdir_path, img_localname)
    if os.path.exists(local_path):
        img_prefix = _gen_prefix()
        img_localname = '%s.%s' % (img_prefix, img_type)
        local_path = os.path.join(subdir_path, img_localname)
        if os.path.exists(local_path):
            raise Exception('%s image name existed' % local_path)

    try:
        # Save the image
        try:
            img_new.save(local_path, quality=quality)
        except IOError:
            img_new.convert('RGB').save(local_path, quality=quality)
    except Exception, err:
        raise Exception('Image file saving failed: %s' % err)

    return local_path


def qrcode_detect(filename):
    """
    Detect if there is QRcode in given image
    :param img: image object
    :return: boolean value if QRcode is existed
    """

    try:
        # Fetch image stream from response
        fp = open(filename, 'rb')
        content = fp.read()
        fp.close()
    except Exception, err:
        raise Exception('%s opening image file failed: %s' % (filename, err))

    try:
        img_detect = np.asarray(bytearray(content), dtype='uint8')
        img = cv2.imdecode(img_detect, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gb = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gb, 100, 200)

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
            if c >= 5:
                found.append(i)

        if len(found) == 3 and _variance(found) > 20:
            return True
        else:
            return False
    except Exception, err:
        raise Exception('Detecting image file %s failed: %s' % (filename, err))


def _get_sized_img(img, content, width, height, face_detect=0):
    """
    Croping image according to assigned width and height
    :param img: image object
    :param content: image content
    :param width: cropped image width
    :param height: cropped image height
    :param face_detect: whether enable face detection
    :return: cropped image object
    """
    if not img:
        raise Exception('empty img object')
    if width > 0 and height > 0:
        img_w = img.size[0]
        img_h = img.size[1]
        ratio = float(width) / float(height)
        if img_w > img_h * ratio:
            center = [img_w / 2, img_h / 2]
        else:
            center = [img_w / 2, 0]

        if face_detect:
            try:
                # Read image as numpy array
                img_detect = np.asarray(bytearray(content), dtype='uint8')
                img_detect = cv2.imdecode(img_detect, cv2.IMREAD_COLOR)

                if face_detect:
                    center = _crop_center(img_detect)
            except Exception, err:
                raise Exception('detection exception: %s' % err)

        # Calculate the width/height ratio of image

        left = 0
        right = img_w
        top = 0
        bottom = img_h

        # Crop image according to the ratio
        if img_w > img_h * ratio:
            left = center[0] - int(img_h * ratio / 2)
            if left < 0:
                left = 0
            elif left > img_w - int(img_h * ratio):
                left = img_w - int(img_h * ratio)
            right = left + int(img_h * ratio)
        else:
            top = center[1] - int(img_w / (ratio * 2))
            if top < 0:
                top = 0
            elif top > img_h - int(img_w / ratio):
                top = img_h - int(img_w / ratio)
            bottom = top + int(img_w / ratio)

        img_new = img.crop((left, top, right, bottom))
        img_new = img_new.resize((width, height), Image.ANTIALIAS)
    else:
        img_new = img
    return img_new


def _crop_center(img):
    """
    Calculate the cropping center by face detecting
    :param img: image object
    :return: coordination of cropping center: [x, y]
    """
    # Create haar cascade
    face_cascade = cv2.CascadeClassifier(_casc_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(15, 15),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    center = [0, 0]
    point_num = len(faces)
    if point_num == 0:
        return center
    for (x, y, w, h) in faces:
        center[0] += (x + w/2)
        center[1] += (y + h/2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # center[0] /= point_num
    # center[1] /= point_num

    cv2.rectangle(img, (center[0], center[1]), (center[0] + 5, center[1] + 5), (0, 0, 255), 2)
    cv2.imshow("Faces found", img)
    cv2.waitKey(0)
    return center


def _variance(data):
    """
    Calculate variance for a list of number
    :param data: list of number
    :return: float variance
    """
    e = 0
    d = 0
    n = len(data)
    for x in data:
        e += x
    e /= float(n)
    for x in data:
        d += math.pow((x - e), 2)
    d = math.sqrt(d / float(n))
    return d


def _gen_prefix():
    """
    Generate a random prefix for image file
    :return: string prefix
    """
    timestamp = str(int(time.time() * 1000))[-5:-1]
    code = ''.join([str(random.randint(0, 9)) for x in xrange(0, 4)])
    prefix = '%s%s' % (timestamp, code)
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image cropper')
    result = {}
    try:
        parser.add_argument('-i', action='store', required=True,
                            dest='input',
                            help='Url or filename of source image')
        parser.add_argument('-w', action='store', default=0, type=int,
                            dest='width',
                            help='Width of cropped image')
        parser.add_argument('-l', action='store', default=0, type=int,
                            dest='height',
                            help='Height of cropped image')
        parser.add_argument('-t', action='store', default='jpg',
                            dest='type',
                            help='File type of cropped image, default is jpg')
        parser.add_argument('-q', action='store', default=100, type=int,
                            dest='quality',
                            help='Quality of cropped image, 1 to 100')
        parser.add_argument('-f', action='store_true', default=False,
                            dest='face_detect',
                            help='Enable face detection')
        parser.add_argument('-m', action='store', default='url',
                            dest='mode',
                            help='Imaging mode: url | file | qr')
        result = parser.parse_args(sys.argv[1:])
    except Exception, err:
        print('Params error: %s' % err)
        exit(1)

    try:
        mode = result.mode
        source = result.input
        if mode == 'qr':
            is_qrcode = qrcode_detect(source)
            print('Image did%s contain QR code' % (' not' * (not is_qrcode)))
        elif mode == 'file':
            local_path = crop_image_file(source,
                                         width=result.width,
                                         height=result.height,
                                         img_type=result.type,
                                         quality=result.quality,
                                         face_detect=result.face_detect)
            print('Cropped image file successfully! Image path is %s' % local_path)
        elif mode == 'url':
            local_path = crop_image(source,
                                    width=result.width,
                                    height=result.height,
                                    img_type=result.type,
                                    quality=result.quality,
                                    face_detect=result.face_detect)
            print('Cropped remote image successfully! Image path is %s' % local_path)
        else:
            raise Exception('Unhandled args')
    except Exception, err:
        print(err)
