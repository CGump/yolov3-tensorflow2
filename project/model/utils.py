# -*- coding: utf-8 -*-
# ================================================================
#
#   Editor      : PyCharm
#   File name   : utils.py
#   Author      : CGump
#   Email       : huangzhigang93@gmail.com
#   Created date: 2020/6/27 15:21
#
# ================================================================
import numpy as np
from functools import reduce
from PIL import Image, ImageEnhance
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """
    用于构建网络层的组合，输入网络层由左向右计算；
    输入不可以为空。
    :param funcs: 网络层
    :return: 网络层组合函数的**引用**
    """
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def letterbox_image(image, size):
    """
    使用填充的方式，在原始图像宽高比不变的情况下调整图像大小
    测试用
    :param image:PIL.JpegImagePlugin.JpegImageFile
    :param size:调整后的图像大小
    :return:调整后的图像
    """
    iw, ih = image.size  # 原始图像的尺寸    w, h = size  # 原始图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', size, (128, 128, 128))  # 新建一个灰色的RGB图像画布
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将原图像放置于画布中心处
    return new_image


def rand(a=0, b=1):
    """
    随机函数
    :param a: 随机量1
    :param b: 随机量2
    :return: 随机抖动值
    """
    return np.random.rand()*(b-a) + a


# todo
def image_enhance():
    pass


# todo
def color_jitter():
    pass


# todo
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    """
    实时数据增强，用于网络训练时批batch图像的预处理，分为平移变换、图像增强、颜色抖动三种
    :param annotation_line:由训练文件（2007_train.txt)按行读入的训练图像名文件（含后缀）
    :param input_shape:图像输入网络的尺寸，一般为（416，416，3）
    :param random:
    :param max_boxes:
    :param jitter:
    :param hue:
    :param sat:
    :param val:
    :param proc_img:
    :return:
    """
    line = annotation_line.split()  # 训练文件每行格式为文件名 x,y,w,h,lable x,y,w,h,lable ……
    image = Image.open(line[0])  # 读取图像文件
    if image.mode != 'RGB':
        image = image.convert('RGB')  # 将图片强制转化为RGB格式
    iw, ih = image.size
    h, w = input_shape


