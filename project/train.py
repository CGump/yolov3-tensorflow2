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
CLASSES_PATH = cfg.get("share", "classes_path")  # 类别文件voc_classes.txt存放地址
ANCHORS_PATH = cfg.get("share", "anchors_path")  # 锚框文件yolo_anchors.txt存放地址
INPUT_WIGHT = cfg.getint("share", "input_wight")  # 输入图像宽
INPUT_HEIGHT = cfg.getint("share", "input_height")  # 输入图像高
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


def data_generator(annotation_line, batch_size, input_shape, anchors, num_classes):
    """
    训练数据生成器
    :param annotation_line: 图片名与标注
    :param batch_size: 批大小
    :param input_shape: 输入尺寸
    :param anchors: 锚框坐标
    :param num_classes: 类别数
    :return: 生成器，输出每个batch的image data和y_true
    """
    n = len(annotation_line)
    if n == 0 or batch_size <= 0:
        return None
    else:
        i = 0
        while True:
            image_data = []
            box_data = []
            for batch in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_line)  # 对图片进行乱序
                # 对每张图片进行数据增强，并分开图片地址和标签
                image, box = get_random_data(annotation_line[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)


def create_model(input_shape, anchors, num_classes, ignore_thresh, load_pretrained=True, freeze_body=2,
                 weights_path='data/yolo_weights.h5'):
    """
    创建训练模型
    :param input_shape: 输入层尺寸
    :param anchors: 锚框坐标
    :param num_classes: 类别数
    :param ignore_thresh: iou阈值
    :param load_pretrained: 预训练控制位
    :param freeze_body: 冻结控制层数
    :param weights_path: 预训练模型地址
    :return: 创建的模型包括模型主体和损失函数
    """
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    # [(13, 13, 3, n+6), (26, 26, 3, n+6), (52, 52, 3, n+6)]三种大小
    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[layer], w//{0: 32, 1: 16, 2: 8}[layer],
                           num_anchors//3, num_classes+5)) for layer in range(3)]  # 这里的3可以按三种尺度的锚框数来统计
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print(f"Creat YOLOv3 model with {num_anchors} anchors and {num_classes} classes.")
    # 加载预训练，并冻结非输出层
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"Load weights {weights_path}")
        if freeze_body in [1, 2]:
            # 冻结除了最后三层外的所有层
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print(f"Freeze the frist {num} layers of total {len(model_body.layers)} layers")
    # 构建损失层，计算损失
    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss",
                        arguments={"anchors": anchors, "num_classes": num_classes, "ignore_thresh": ignore_thresh})(
                        [*model_body.output, *y_true])

    models = Model([model_body.input, *y_true], model_loss)
    return models


#  ==============================  训练进行  ================================
if __name__ == '__main__':
    class_name = get_classes(CLASSES_PATH)
    anchor = get_anchors(ANCHORS_PATH)
    num_class = len(class_name)
    # 日志记录
    logger.info(f"Train settings —— classes num: {num_class}, input shape: {INPUT_SHAPE}, val aplit: {VAL_SPLIT} \n \
                freeze: {FREEZE}, freeze batch: {BATCH_FREEZE}, freeze lr: {LEAENING_RATE_FREEZE}, \n \
                freeze epochs: {EPOCH_FREEZE} batch size: {BATCH_YOLO}, learning rate: {LEARNING_RATE_YOLO}, \n \
                epochs: {EPOCH_YOLO}, ignore thresh: {IOU_THRESH}")
    # 创建模型
    model = create_model(input_shape=INPUT_SHAPE, anchors=anchor, num_classes=num_class, ignore_thresh=IOU_THRESH,
                         load_pretrained=PRETRAINED, freeze_body=2, weights_path=WEIGHT_PATH)
    # 设置回调函数
    logging = TensorBoard(log_dir=LOG_DIR)
    # 该回调函数将在每个epoch后保存模型到log_dir
    checkpoint = ModelCheckpoint(LOG_DIR + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # 当评价指标不再提升时，减少学习率
    reduce_learingrate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # 当监测值不再改善时，该回调函数将终止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 划归训练集和验证集
    with open(ANNATION_PATH) as file:
        lines = file.readlines()
    np.random.seed(10101)  # 保证每次生成的随机序列一致
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * VAL_SPLIT)
    num_train = len(lines) - num_val
    # 冻结训练
    if FREEZE:
        model.compile(optimizer=Adam(lr=LEAENING_RATE_FREEZE), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print(f"Train on {num_train} samples, val on {num_val} samles, with batch size {BATCH_FREEZE}")
        model.fit_generator(data_generator(lines[:num_train], BATCH_FREEZE, INPUT_SHAPE, anchor, num_class),
                            steps_per_epoch=max(1, num_train//BATCH_FREEZE),
                            validation_data=data_generator(lines[num_train:],
                                                           BATCH_FREEZE, INPUT_SHAPE, anchor, num_class),
                            validation_steps=max(1, num_val//BATCH_FREEZE),
                            epochs=EPOCH_FREEZE,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(f"{LOG_DIR}trained_weights_stage_1.h5")

    # 解冻，整体训练
    if YOLO:
        for s in range(len(model.layers)):
            model.layers[s].trainable = True
        # 重新编译以适配改动
        model.compile(optimizer=Adam(lr=LEARNING_RATE_YOLO), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print("Unfreeze all of the layers.")
        print(f"Train on {num_train} samples, val on {num_val} samples, with batch size {BATCH_YOLO}.")
        model.fit_generator(data_generator(lines[:num_train], BATCH_YOLO, INPUT_SHAPE, anchor, num_class),
                            steps_per_epoch=max(1, num_train//BATCH_YOLO),
                            validation_data=data_generator(lines[num_train:],
                                                           BATCH_YOLO, INPUT_SHAPE, anchor, num_class),
                            validation_steps=max(1, num_val//BATCH_YOLO),
                            epochs=EPOCH_YOLO,
                            initial_epoch=EPOCH_FREEZE,  # 这里设置是起点位控制，如果不冰冻训练那么就是0
                            callbacks=[logging, checkpoint, reduce_learingrate, early_stopping])
        model.save_weights(f"{LOG_DIR}trained_weights_final.h5")
