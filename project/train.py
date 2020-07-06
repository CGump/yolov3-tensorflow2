# -*- coding: utf-8 -*-
# ================================================================
#
#   Editor      : PyCharm
#   File name   : train.py
#   Author      : CGump
#   Email       : huangzhigang93@gmail.com
#   Created date: 2020/7/6 10:37
#
# ================================================================
import os
import time
import logging
import configparser
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.yolo3 import yolo_body, yolo_loss, preprocess_true_boxes
from model.utils import get_random_data, set_logging

#  ==============================   初始化   ================================
LOG_DIR = f"logs/{time.strftime('%Y%m%d_%H')}/"
TRAIN_LOG_NAME = "train.log"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)  # makedirs会自动创建父目录

set_logging(file_path=LOG_DIR + TRAIN_LOG_NAME)
logger = logging.getLogger(__name__)

cfg = configparser.ConfigParser()
cfg.read("model/config.ini", encoding="utf8")


#  ==============================  训练配置  ================================

ANNATION_PATH = cfg.get("train", "pointer_path")  # 训练集指针文件
CLASSES_PATH = cfg.get("train", "classes_path")  # 类别文件voc_classes.txt存放地址
ANCHORS_PATH = cfg.get("train", "anchors_path")  # 锚框文件yolo_anchors.txt存放地址
INPUT_WIGHT = cfg.getint("train", "input_wight")  # 输入图像宽
INPUT_HEIGHT = cfg.getint("train", "input_height")  # 输入图像高
INPUT_SHAPE = (INPUT_WIGHT, INPUT_HEIGHT)  # 输入图像尺寸
PRETRAINED = cfg.getboolean("train", "pretrained")  # 迁移训练控制位
WEIGHT_PATH = cfg.get("train", "weight_path")  # 预训练文件
VAL_SPLIT = cfg.getfloat("train", "val_split")  # 验证集划分比例
FREEZE = cfg.getboolean("train", "freeze")  # 冻结层控制位
LEAENING_RATE_FREEZE = cfg.getfloat("train", "learning_rate_freeze")  # 冻结层学习率
EPOCH_FREEZE = cfg.getint("train", "epoch_freeze") if FREEZE else 0  # 冻结层迭代次数
BATCH_FREEZE = cfg.getint("train", "batch_freeze")  # 冻结层批次大小
YOLO = cfg.getboolean("train", "yolo")  # 整体训练控制位
LEARNING_RATE_YOLO = cfg.getfloat("train", "learning_rate_yolo")  # 整体训练学习率
EPOCH_YOLO = cfg.getint("train", "epoch_yolo")  # 整体训练迭代次数
BATCH_YOLO = cfg.getint("train", "batch_yolo")  # 整体训练批次大小
IOU_THRESH = cfg.getfloat("train", "iou_thresh")  # iou阈值


#  ==============================  训练过程  ================================

def get_classes(classes_path):
    """
    获取类名
    :param classes_path: 类名文件地址
    :return: 类名列表
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """
    获取锚框文件
    :param anchors_path: 锚框文件
    :return: 锚框列表
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
