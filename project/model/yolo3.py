# -*- coding: utf-8 -*-
# ================================================================
#
#   Editor      : PyCharm
#   File name   : yolo3.py
#   Author      : CGump
#   Email       : huangzhigang93@gmail.com
#   Created date: 2020/6/30 15:34
#
# ================================================================
import numpy as np
import tensorflow as tf
from functools import wraps
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from model.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """
    设置yolov3网络中2D卷积层参数（参照Darknet）
    :param args:**
    :param kwargs:**
    :return:Conv2D
    """
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """
    yolov3网络中的“卷积+标准化+激活层”结构：conv -> BN -> activate
    :param args: **
    :param kwargs: **
    :return:
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))


def resblock_body(input_layer, num_filters, num_blocks):
    """
    yolov3网络的残差块结构:conv -> res ->out
    :param input_layer: 网络输入
    :param num_filters: 卷积核数
    :param num_blocks: 残差块结构数
    :return: 输出
    """
    x = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
                    DarknetConv2D_BN_Leaky(num_filters, (3, 3))
                    )(x)
        x = Add()([x, y])  # 残差相加
    return x


def darknet_body(input_layer):
    """
    yolo主干网络的主体，一共具有52个卷积层：1+1*5+(1+2+8+8+4)*5=52
    :param input_layer: 网络输入
    :return: 输出
    """
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(input_layer)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(input_layer, num_filters, out_filters):
    """
    yolov3网络的输出层，由5个卷积层和1个卷积层加1个线性输出层构成，
    其中5个卷积层的输出会连接到下一个输出层
    :param input_layer: 输入
    :param num_filters: 中间层的卷积核数
    :param out_filters: 输出层的卷积核数
    :return: yolo输出x，以及下一层的输入y
    """
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1))
                )(input_layer)
    y = compose(DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1))
                )(x)
    return x, y


# todo
def yolo_body(inputs, num_anchors, num_classes):
    """
    yolov3的卷积神经网络主体
    :param inputs: 输入
    :param num_anchors: 锚框数
    :param num_classes: 类数
    :return: Model
    """
    darknet = Model(inputs, darknet_body(inputs))
    pass


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    model_input = Input(shape=(608, 608, 3))
    # model_output = yolo_body(model_input, 3, 80)
    # model = Model(model_input, model_output)
    # model.summary()
    pass
