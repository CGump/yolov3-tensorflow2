# -*- coding: utf-8 -*-
# ================================================================
#
#   Editor      : PyCharm
#   File name   : yolo_api.py
#   Author      : CGump
#   Email       : huangzhigang93@gmail.com
#   Created date: 2020/7/13 9:33
#
# ================================================================
import os
import json
import colorsys
import numpy as np
import configparser
import tensorflow as tf

from timeit import default_timer as timer
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from model.yolo3 import yolo_eval, yolo_body
from model.utils import letterbox_image

#  ==============================   初始化   ================================
cfg = configparser.ConfigParser()
cfg.read("model/config.ini", encoding="utf8")

MODEL_PATH = cfg.get("test", "model_path")
ANCHORS_PATH = cfg.get("share", "anchors_path")
CLASSES_PATH = cfg.get("share", "classes_path")
SCORE = cfg.getfloat("test", "score")
IOU = cfg.getfloat("test", "iou")
INPUT_SIZE = (cfg.getint("share", "input_wight"), cfg.getint("share", "input_height"))
GPU_NUM = cfg.getint("test", "GPU_num")


#  ==============================   封装   ==================================
class YOLO(object):
    def __init__(self, **kwargs):
        self.model_path = MODEL_PATH
        self.anchors_path = ANCHORS_PATH
        self.classes_path = CLASSES_PATH
        self.score = SCORE
        self.iou = IOU
        self.model_image_size = INPUT_SIZE
        self.gpu_num = GPU_NUM
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = 0

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        """

        :return:
        """
