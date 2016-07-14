#! /usr/bin/python2.7
# -*- coding: utf8 -*-

"""
clean.py
author: danydliu
定期执行脚本，清理redis和本地data、log的过期数据
"""

import datetime
import os
import re
import shutil
import time

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
LOG_PREFIX = ['crop', 'service', 'access']

FILE_DURATION = 3
MERGE_INTERVAL = 1


def cleandir(path, days=FILE_DURATION):
    time_now = time.time()
    for root, dirs, files in os.walk(path):
        for name in dirs:
            mtime = os.stat(os.path.join(root, name)).st_mtime
            if time_now - mtime > 3600 * 24 * days:
                shutil.rmtree(os.path.join(root, name))
        for name in files:
            mtime = os.stat(os.path.join(root, name)).st_mtime
            if time_now - mtime > 3600 * 24 * days:
                os.remove(os.path.join(root, name))


def merge_log(path, prefix, days):
    log_time = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y.%m.%d')
    filename = '%s_%s.log' % (prefix, log_time)
    output = open(os.path.join(path, filename), 'a+')
    # data = ''
    for root, dirs, files in os.walk(path):
        for name in files:
            if re.search(prefix + r'[^_]*_' + log_time + '.+\.log', name):
                try:
                    fp = open(os.path.join(root, name), 'r')
                    data = fp.read()
                    output.writelines(data)
                    fp.close()
                    os.remove(os.path.join(root, name))
                except IOError:
                    pass
    output.close()


if __name__ == '__main__':
    if os.path.exists(DATA_DIR):
        cleandir(DATA_DIR)
    if os.path.exists(LOG_DIR):
        cleandir(LOG_DIR)
        for prefix in LOG_PREFIX:
            merge_log(LOG_DIR, prefix, MERGE_INTERVAL)


