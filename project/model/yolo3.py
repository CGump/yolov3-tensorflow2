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
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
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
        y = compose(DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
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
    其中5个卷积层的输出会连接到下一个输出层,
    这里的结构类似FPN特征金字塔
    :param input_layer: 输入
    :param num_filters: 中间层的卷积核数
    :param out_filters: 输出层的卷积核数
    :return: yolo输出x，以及下一层的输入y
    """
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1))
                )(input_layer)
    y = compose(DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1))
                )(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """
    yolov3的卷积神经网络主体
    :param inputs: 输入
    :param num_anchors: 锚框数
    :param num_classes: 类数
    :return: Model
    """
    darknet = Model(inputs, darknet_body(inputs))
    # 第一个输出13*13的小尺度特征图，对应检索大尺度特征
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                UpSampling2D(2)
                )(x)
    x = Concatenate()([x, darknet.layers[152].output])
    # 第二个输出26*26的中尺度特征图，对应检索中尺度特征
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                UpSampling2D(2)
                )(x)
    x = Concatenate()([x, darknet.layers[92].output])
    # 第三个输出52*52的大尺度特征图，对应检索小尺度特征
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    return Model(inputs, [y1, y2, y3])


def yolo_head(feature, anchors, num_classes, input_shape, calc_loss=False):
    """
    yolov3主干网络至特征与边界框的链接
    :param feature: 网络输出的特征
    :param anchors: ；锚框
    :param num_classes: 类别数
    :param input_shape: 输入尺寸
    :param calc_loss: 训练/测试识别位
    :return: box的坐标、宽高、边界框置信度、类别置信度
    """
    # 预设锚框：10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    num_anchors = len(anchors)
    # 将锚框坐标转换为张量：batch, height, width, num_anchors, box_params
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    # [[[[[ 10.  13.]
    #           ...
    #     [373. 326.]]]]], shape=(1, 1, 1, 9, 2), dtype=float32
    grid_shape = K.shape(feature)[1:3]  # 特征图的宽、高，分别对应的三个尺度输出：(13,13), (26,26), (52,52)
    # K.tile()在维度上复制，K.arange()会创建一个列表迭代
    # 构建一个按输出尺度大小的栅格网格，比如输出是(13,13)，保证网格坐标从(0,0)~(13,13)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feature))
    feature = K.reshape(feature, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # 将box_xy, box_wh从output的预测数据转为标准尺度的坐标
    box_xy = (K.sigmoid(feature[..., :2] + grid) / K.cast(grid_shape[::-1], K.dtype(feature)))
    box_wh = K.exp(feature[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feature))
    box_confidence = K.sigmoid(feature[..., 4:5])
    box_class_probs = K.sigmoid(feature[..., 5:])

    if calc_loss:
        return grid, feature, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_box(box_xy, box_wh, input_shape, image_shape):
    """
    校正预测框
    :param box_xy: 同上box_xy
    :param box_wh: 同上box_wh
    :param input_shape: 输入图像尺寸
    :param image_shape: 原图尺寸
    :return:
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))  # 强制转换格式
    image_shape = K.cast(image_shape, K.dtype(box_yx))  # 强制转换格式
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))  # 求取输入/原图尺寸的最小值，四舍五入取整获得新图像的尺寸
    offset = (input_shape - new_shape) / 2. / input_shape  # 补偿值
    scale = input_shape / new_shape  # 收缩尺度
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxs = box_yx + (box_hw / 2.)
    boxes = K.concatenate([box_mins[..., 0:1],  # y_min
                           box_mins[..., 1:2],  # x_min
                           box_maxs[..., 0:1],  # y_max
                           box_maxs[..., 1:2]  # x_max
                           ])
    # 将预测框缩放到原始图像形状
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feature, anchors, num_classes, input_shape, image_shape):
    """
    整合yolo的卷积输出
    :param feature: 特征输出
    :param anchors: 锚框
    :param num_classes: 类别数
    :param input_shape: 输入尺寸
    :param image_shape: 图像尺寸
    :return: 预选框，分类置信度
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feature, anchors, num_classes, input_shape)
    boxes = yolo_correct_box(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs  # 置信度计算
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=0.6, iou_threshold=0.5):
    """
    在yolov3模型的输出结果中进行预测，返回目标框、置信度和类别预测
    :param yolo_outputs: yolo网络输出
    :param anchors: 锚框
    :param num_classes: 类别数
    :param image_shape: 输入图像尺寸
    :param max_boxes: 最大目标框数
    :param score_threshold: 置信度阈值
    :param iou_threshold: IOU阈值
    :return: 目标框、置信度和类别预测
    """
    num_layers = len(yolo_outputs)  # yolov3的特征输出[y1, y2, y3]
    # 由于没有用tiny_yolo，所以直接给出9个锚框的数值，如果需要tiny yolo则需要加上if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32  # 这里使用小尺度来还原输出
    boxes = []
    box_scores = []
    for layer in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[layer],  # 选取yolo输出层
                                                    anchors[anchor_mask[layer]],  # 根据输出层排布选取锚框
                                                    num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 降维
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold  # 根据置信度阈值进行筛选，返回的是布尔张量
    max_boxes_tensor = K.constant(max_boxes, dtype=tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 清理box中所有不达标的目标框
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 非极大值抑制，最多只能有max_boxes个目标框
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold)
        # 通过下标找到box和对应的分数
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, dtype=tf.int32)
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes: int):
    """
    将标注框转化为训练输入格式
    :param true_boxes: 数组，(m, t, 5)，包括x_min, y_min, x_max, y_max, class_id
    :param input_shape: 输入的宽高，必须是32的倍数（也必须是416的倍数）
    :param anchors: 数组，(9, 2)
    :param num_classes: 整数
    :return: y_true，真实值
    """
    # 此断言出判断是否所有的class id都小于总类数
    assert (true_boxes[..., 4] < num_classes).all(), "class id must be less than num_classes"
    num_layers = len(anchors) // 3  # 锚框数都是3个一组，这里anchors为9
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # if num_layers==3 else [[3,4,5], [1,2,3]]
    true_boxes = np.array(true_boxes, dtype='float32')  # shape: (图片张数, 每张图片的box数， 5)
    input_shape = np.array(input_shape, dtype='int32')  # shape: (2,), [416, 416]
    # 将每个box的xmin，ymin和xmax，ymax分别相加后除以2，即为去中心点坐标
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # 计算每个box的宽高值：xmax-xmin，ymax-ymin
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # 分别除以416
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]  # 分别除以416

    m = true_boxes.shape[0]  # 图片的张数
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[i] for i in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[i][0], grid_shapes[i][1], len(anchor_mask[i]), 5 + num_classes),
                       dtype='float32') for i in range(num_layers)]
    # anchors扩维并将位置移动到x轴上下
    anchors = np.expand_dims(anchors, 0)
    anchor_max = anchors / 2.
    anchor_min = -anchor_max
    valid_mask = boxes_wh[..., 0] > 0

    for i in range(m):
        wh = boxes_wh[i, valid_mask[i]]  # 去除全0行
        if len(wh) == 0:
            continue
        wh = np.expand_dims(wh, -2)
        box_max = wh / 2.
        box_min = -box_max

        # 计算真实值和锚框的IOU
        intersect_min = np.maximum(box_min, anchor_min)
        intersect_max = np.minimum(box_max, anchor_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 通过iou最大的确定box应该放在label的那个anchor位置
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for layer in range(num_layers):
                if n in anchor_mask[layer]:  # 检查best anchor在哪一个尺度
                    e = np.floor(true_boxes[i, t, 0] * grid_shapes[layer][1]).astype('int32')
                    j = np.floor(true_boxes[i, t, 1] * grid_shapes[layer][0]).astype('int32')
                    k = anchor_mask[layer].index(n)
                    c = true_boxes[i, t, 4].astype('int32')

                    y_true[layer][i, j, e, k, 0:4] = true_boxes[i, t, 0:4]
                    y_true[layer][i, j, e, k, 4] = 1
                    y_true[layer][i, j, e, k, 5 + c] = 1
    # y_true:[(None,13,13,3,5+num_classes),(None,26,26,3,5+num_classes),(None,52,52,3,5+num_classes)]
    return y_true


# todo
def box_iou(b1, b2):
    """
    计算两个目标框的IOU
    :param b1: 第一个框
    :param b2: 第二个框
    :return: iou值张量
    """
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_min = b1_xy - b1_wh_half
    b1_max = b1_xy + b1_wh_half

    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_min = b2_xy - b2_wh_half
    b2_max = b2_xy + b2_wh_half

    intersect_min = K.maximum(b1_min, b2_min)
    intersect_max = K.minimum(b1_max, b2_max)
    intersect_wh = K.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=0.5, print_loss=False):
    """
    yolov3的loss计算，中心点loss、宽高loss、置信度loss、类别回归loss
    :param args: list的组合，包括了预测值和真实值，具体为：
                 arg[:num_layers]--预测值yolo_outputs,
                 arg[num_layers:]--真实值y_true
    :param anchors: 锚框数
    :param num_classes: 类别数
    :param ignore_thresh: iou阈值，忽略小于该阈值的目标框
    :param print_loss: loss的打印开关
    :return: loss张量
    """
    num_layers = len(anchors) // 3  # 计算输出层数，一般是3层，yolo—tiny是2层
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))  # 很显然，这里计算输入尺寸默认(416, 416)
    grid_shapes = [K.cast(K.shape(yolo_outputs[layer])[1:3], K.dtype(y_true[0])) for layer in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # m是batch size
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for layer in range(num_layers):
        object_mask = y_true[layer][..., 4:5]  # 置信度
        true_class_probs = y_true[layer][..., 5:]  # 分类
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[layer], anchors[anchor_mask[layer]],
                                                     num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])  # 相对于gird的box参数(x, y, w, h)
        # 对x, y, w, b转换公式的反变换
        raw_true_xy = y_true[layer][..., :2] * grid_shapes[layer][::-1] - grid  # :2就是x,y
        raw_true_wh = K.log(y_true[layer][..., 2:4] / anchors[anchor_mask[layer]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # 避免log(0)=-inf的情况
        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4]

        # 遍历每个batch，这里tf.TensorArray相当于一个动态数组，size=1表示为二维
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')  # 将真实标定的数据转换为T or F的掩膜

        def loop_body(b, ignore_masks):  # 这个ignore_mask很有意思 #todo
            # object_mask_bool(b, 13, 13, 3, 4)--五维数组，第b张图的第layer层feature map
            # true_box将第b图第layer层feature map,有目标窗口的坐标位置取出来。true_box[x,y,w,h]
            true_box = tf.boolean_mask(y_true[layer][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)  # 计算预测值和真实的iou
            best_iou = K.max(iou, axis=-1)  # 取每个grid上多个anchor box的最大iou
            ignore_masks = ignore_masks.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_masks

        _, ignore_mask = tf.while_loop(lambda b: b < m, loop_body, [0, ignore_mask])  # todo
        ignore_mask = ignore_mask.stack()  # 将一个列表的维数数目为R的张量堆积起来形成R+1维新张量，这里R应该是b
        ignore_mask = K.expand_dims(ignore_mask, -1)  # ignoer_mask的shape是(b, 13, 13, 3, 1) "13 13"有三个layer
        # x,y的交叉熵损失，这里ignore_mask确实还没怎么弄明白，后面还需要看看
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy,
                                                                       raw_pred[..., 0:2],
                                                                       from_logits=True)
        # w,h的均方差损失
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        # 置信度交叉熵损失，这里没有物体的部分也要计算损失，因此是object和1-object的和
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
            (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        # 分类损失
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        # 计算一个batch的总损失
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss:')
    return loss


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    model_input = Input(shape=(1248, 1248, 3))
    model_output = yolo_body(model_input, 3, 80)
    model_output.summary()
