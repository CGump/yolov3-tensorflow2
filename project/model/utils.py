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
    w, h = size
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', size, (128, 128, 128))  # 新建一个灰色的RGB图像画布
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将原图像放置于画布中心处
    return new_image


def rand(a=0.0, b=1.0):
    """
    随机函数
    :param a: 随机量1
    :param b: 随机量2
    :return: 随机抖动值
    """
    return np.random.rand() * (b - a) + a


def image_enhance(image):
    """
    图像增强，包括饱和度、亮度、对比度、锐度
    :param image: 输入图像
    :return: 增强后图像
    """
    random_factor = np.random.randint(5, 16) / 10.  # 饱和度在0.5~1.5之间合适
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(8, 13) / 10.  # 亮度在0.8~1.8之间合适
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(8, 16) / 10.  # 对比度在0.8~1.5之间合适
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(10, 50) / 10.  # 锐度在1.~5.之间合适
    image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    return image


# todo
def color_jitter(image):
    """
    颜色抖动，将RGB色域图像转换为HSV色域，然后随机抖动
    :param image:输入图像
    :return:颜色抖动后图像
    """
    hue, sat, val = (0.1, 1.5, 1.5)
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x * 255)  # numpy array, 0 to 1
    return image


# todo
def get_random_data(annotation_line, input_shape, max_boxes=20, jitter=0.3,
                    random=True, data_enhance=False, dithering=False):
    """
    实时数据增强，用于网络训练时批batch图像的预处理，分为平移变换、图像增强、颜色抖动三种
    :param annotation_line:由训练文件（2007_train.txt)按行读入的训练图像名文件（含后缀）
    :param input_shape:图像输入网络的尺寸，一般为（416，416，3）
    :param max_boxes:最大标注框数，这里没写好，应该是由输入决定的
    :param jitter:宽高比随机系数
    :param random:随机增强控制位，默认True
    :param data_enhance:数据增强控制位，默认False
    :param dithering:颜色抖动控制位，默认False
    :return:image_data--归一化后图像，box_data--调整后box，有20个list
    """
    line = annotation_line.split()  # 训练文件每行格式为文件名 xmin,ymin,xmax,ymax,lable  xmin,ymin,xmax,ymax,lable ……
    image = Image.open(line[0])  # 读取图像文件
    if image.mode != 'RGB':
        image = image.convert('RGB')  # 将图片强制转化为RGB格式
    iw, ih = image.size
    h, w = input_shape
    # 从每张图像的annotation_line中取出每一个标注框的xmin,ymin,xmax,ymax,lable，按行分布
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # 随机数据增强标志位为False的情况下，只进行图像的resize
    # 以图像填充的方式使得输入图像等比例缩小在(416, 416)大小的画布上、
    # 同理，标注框的坐标位置和尺寸也随着图像的缩小进行调整
    if not random:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255.

        box_data = np.zeros(shape=(max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)  # 打乱标注框的顺序
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx  # 水平方向
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy  # 垂直方向
            box_data[:len(box)] = box
        return image_data, box_data

    # 生成随机的宽高比
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # 随机水平位移
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 随机水平翻转
    flip = rand() < 0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 随机数据增强
    if data_enhance:
        image = image_enhance(image)

    # 随机颜色抖动：RGB->HSV+抖动->RGB
    if dithering:
        image = color_jitter(image)

    # 归一化
    image_data = np.array(image) / 255.

    # 校正标注框
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        # 缩放，跟随图像比例改变
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        # 左右翻转
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        # 计算新长宽
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        # 丢弃无效的box
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

        return image_data, box_data


if __name__ == '__main__':
    # todo:完成图像resize的预览，包括标注和标签
    from matplotlib import pyplot as plt
    from PIL import ImageDraw

    test_list = ["F:/my-learning/yolov3-tensorflow2/doc/img/apple.jpg 876,331,1766,1216,0 123,446,838,1113,0"]
    show_image = Image.open("F:/my-learning/yolov3-tensorflow2/doc/img/apple.jpg")
    test_image, test_box = get_random_data(test_list[0], input_shape=(416, 416), random=False)
    test_image = np.array(test_image * 255, dtype=np.int)
    test_box = test_box[0:2]
    test_image = np.uint8(test_image)
    test_image = Image.fromarray(test_image)

    draw1 = ImageDraw.Draw(show_image)
    draw1.rectangle([876, 331, 1766, 1216], outline=(0, 255, 0), width=5)
    draw1.rectangle([123, 446, 838, 1113], outline=(0, 255, 0), width=5)

    draw2 = ImageDraw.Draw(test_image)
    for b in test_box:
        draw2.rectangle(list(b[..., 0:4]), outline=(255, 0, 0), width=5)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(show_image)
    plt.subplot(122)
    plt.imshow(test_image)
    plt.show()
