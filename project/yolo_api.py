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

from timeit import default_timer as timer
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from model.yolo3 import yolo_eval, yolo_body
from model.utils import letterbox_image
