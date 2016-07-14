# -*- coding: utf8 -*-

"""
imaging.py
author: danydliu
图片裁剪和分发
"""

import httplib2
import commands
import cStringIO
import cv2
import datetime
from log_manager import LogManager
import math
import numpy as np
import os
from PIL import Image
import random
import re
import time
import urllib


_log_manager = LogManager('log')
_logger_crop = _log_manager.logger_crop
_casc_path = 'haarcascade_frontalface_default.xml'


def crop_image(url, width, height, img_type='jpg', quality=100, **kwargs):
    """
    :param url: 源图片地址
    :param width: 裁剪宽度
    :param height: 裁剪高度
    :param img_type: 图片类型
    :param quality: 压缩质量
    :return:
    """
    face_detect = kwargs.get('face_detect')
    qr_detect = kwargs.get('qr_detect')
    site = 'http://img1.gtimg.com'
    root_path = '/rcdimg'
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    h = httplib2.Http('.cache')
    proto, rest = urllib.splittype(url)
    host, path = urllib.splithost(rest)
    result = {}
    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:5.0) Gecko/20110619 Firefox/5.0'
    }
    if host:
        headers['Host'] = host
    try:
        (resp, content) = h.request(url, 'GET', headers=headers)
        if content[0:20].find('The requested URL') != -1:
            raise httplib2.HttpLib2Error('URL cannot find in server')
    except httplib2.HttpLib2Error, err:
        _logger_crop.error('%s request error: %s' % (url, str(err)))
        return result

    result[url] = {}
    try:
        # Fetch image stream from response
        img = Image.open(cStringIO.StringIO(content))
    except Exception, err:
        _logger_crop.error('%s cropping failed: %s' % (url, err))
        return result

    try:
        img_new = _get_sized_img(img, content, width, height, qr_detect, face_detect)
    except Exception, err:
        _logger_crop.error('Sizing image_url: %s error: %s' % (url, err))
        return result

    # Generate a random prefix for image file
    img_prefix = _gen_prefix()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(data_path, dir_prefix.strftime('%Y%m%d'))
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
            _logger_crop.warn('%s image name existed' % local_path)
            return result

    img_destname = '%s_%dx%d.%s' % (img_prefix, width, height, img_type)
    dest_path = '%s/%s/%s/%s' % (root_path, dir_prefix.strftime('%Y%m%d'),
                                 dir_prefix.strftime('%H'),
                                 img_destname)

    try:
        # Save the image
        try:
            img_new.save(local_path, quality=quality)
        except IOError:
            img_new.convert('RGB').save(local_path, quality=quality)

        # Dispatch image file
        _octo_sendfile(dest_path, local_path)
        result[url]['%dx%d' % (width, height)] = site + dest_path
    except Exception, err:
        _logger_crop.error('%s saving or dispatching failed: %s' % (url, err))

    return result


def crop_image_file(source, width, height, img_type='jpg', quality=100, **kwargs):
    """
    :param stream: 图片文件流
    :param width: 裁剪宽度
    :param height: 裁剪高度
    :param img_type: 图片类型
    :param quality: 压缩质量
    :return:
    """
    face_detect = kwargs.get('face_detect')
    qr_detect = kwargs.get('qr_detect')
    site = 'http://img1.gtimg.com'
    root_path = '/rcdimg'
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    result = {}
    try:
        # Fetch image stream from response
        img = Image.open(cStringIO.StringIO(source))
    except Exception, err:
        _logger_crop.error('file image cropping failed: %s' % err)
        return result

    try:
        img_new = _get_sized_img(img, source, width, height, qr_detect, face_detect)
    except Exception, err:
        _logger_crop.error('Sizing image error: %s' % err)
        return result

    # Generate a random prefix for image file
    img_prefix = _gen_prefix()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(data_path, dir_prefix.strftime('%Y%m%d'))
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
            _logger_crop.warn('%s image name existed' % local_path)
            return result

    if width == 0 or height == 0:
        img_destname = '%s.%s' % (img_prefix, img_type)
    else:
        img_destname = '%s_%dx%d.%s' % (img_prefix, width, height, img_type)

    dest_path = '%s/%s/%s/%s' % (root_path, dir_prefix.strftime('%Y%m%d'),
                                 dir_prefix.strftime('%H'),
                                 img_destname)

    try:
        # Save the image
        try:
            img_new.save(local_path, quality=quality)
        except IOError:
            img_new.convert('RGB').save(local_path, quality=quality)

        # Dispatch image file
        _octo_sendfile(dest_path, local_path)
        result = site + dest_path
    except Exception, err:
        _logger_crop.error('file image saving or dispatching failed: %s' % err)

    return result


def _get_sized_img(img, content, width, height, qr_detect, face_detect):
    """
    根据比例裁剪并压缩图片
    :param img: 图片对象
    :param content: 图片内容
    :param width: 宽度
    :param height: 高度
    :param qr_detect: 是否启用二维码识别
    :param face_detect: 是否启用人脸识别
    :return: 处理后的img对象
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

        if qr_detect or face_detect:
            try:
                # Read image as numpy array
                img_detect = np.asarray(bytearray(content), dtype='uint8')
                img_detect = cv2.imdecode(img_detect, cv2.IMREAD_COLOR)

                if qr_detect and qrcode_detect(img_detect):
                    raise Exception('QRCode is found.')
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
            # left = int((img_w - img_h * ratio) / 2)
            left = center[0] - int(img_h * ratio / 2)
            if left < 0:
                left = 0
            elif left > img_w - int(img_h * ratio):
                left = img_w - int(img_h * ratio)
            right = left + int(img_h * ratio)
        else:
            # top = int((img_h - img_w / ratio) / 2)
            # bottom -= top

            top = center[1] - int(img_w / (ratio * 2))
            if top < 0:
                top = 0
            elif top > img_h - int(img_w / ratio):
                top = img_h - int(img_w / ratio)
            bottom = top + int(img_w / ratio)

            # bottom -= int(img_h - img_w / ratio)
        img_new = img.crop((left, top, right, bottom))

        # Thumbnail the image
        # img_new.thumbnail((width, height))

        img_new = img_new.resize((width, height), Image.ANTIALIAS)
    else:
        img_new = img
    return img_new


def _crop_center(img):
    """
    获取人脸识别后的图片裁剪中心
    :param img_content: 图片文件流
    :return: 计算后的裁剪中心
    """
    # Create haar cascade
    face_cascade = cv2.CascadeClassifier(_casc_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
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

    center[0] /= point_num
    center[1] /= point_num

    # cv2.rectangle(img, (center[0], center[1]), (center[0] + 5, center[1] + 5), (0, 0, 255), 2)
    # cv2.imshow("Faces found", img)
    # cv2.waitKey(0)
    return center


def qrcode_detect(img):
    """
    检测图片中是否有二维码
    :param img: 原图片
    :return: 真值
    """
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


def _variance(data):
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
    timestamp = str(int(time.time() * 1000))[-5:-1]
    code = ''.join([str(random.randint(0, 9)) for x in xrange(0, 4)])
    prefix = '%s%s' % (timestamp, code)
    return prefix


def _octo_sendfile(dest, local):
    ip = '127.0.0.1'
    port = 21821
    bin_path = '/usr/local/file_dispatcher/bin/octoSend'
    file_type = 1
    site = 'img1'

    if not (dest and local):
        raise Exception('Dest and local filename must not be None')

    status, cmd = commands.getstatusoutput(
        '%s %s %d %d %s %s %s' % (bin_path, ip, port, file_type, site, dest, local)
    )
    # status, cmd = commands.getstatusoutput(
    #     '/usr/local/file_dispatcher/bin/octoSend 127.0.0.1 21821 1 img1 /rcdimg/cropped.jpg /data/rcdimg/test_img.jpg')
    if str(status) == '0':
        ret = re.sub('\s+', ' ', cmd).split(' ')[1]
        if ret != '0':
            raise Exception('Send file failed, ret code is ' + ret)
    else:
        raise Exception('Octosend cmd exec failed')


if __name__ == '__main__':
    print(crop_image('http://img1.gtimg.com/news/pics/hv1/21/77/2098/136442106.jpg',
                     160, 90, face_detect=1))
    # print(crop_image_file(source='http://www.weixinju.com/uploadfile/2012/1206/20121206104941268.jpg',
    #                       img_type='jpg',
    #                       height=0,
    #                       width=0))

