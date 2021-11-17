import cv2
video_f = ''
mp4 = cv2.VideoCapture("/home/yzi/Proposal/code/ice_detection/data/video/ice_detection_origin_video.mp4")  # 读取视频
is_opened = mp4.isOpened()  # 判断是否打开
print(is_opened)
fps = mp4.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
print(fps)
widght = mp4.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取视频的宽度
height = mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取视频的高度
print(str(widght) + "x" + str(height))
i = 0
while is_opened:
    if i == 1470:  # 截取前10张图片
        break
    else:
        i += 1
    (flag, frame) = mp4.read()  # 读取图片
    file_name = "/home/yzi/Proposal/code/ice_detection/data/images/" + "img" + str(i) + ".jpg"
    print(file_name)
    if flag == True:
        cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY])  # 保存图片
print("转换完成")
# %% crop image
import os
import cv2
import numpy as np

def crop_img(img, crop_bondary):
    img_crop = img[crop_bondary[0]:crop_bondary[0] + crop_bondary[2], crop_bondary[1]:crop_bondary[1] + crop_bondary[3], :]
    return img_crop

def batch_crop_img(img_dir, img_save, crop_bondary):
    EXTENSION = '.jpg'
    imgs_full = os.listdir(img_dir)
    print('Processing ' + str(len(imgs_full)) + ' number of images in the current folder')
    for img in imgs_full:
        # Read images using OpenCV and normalize into grayscale
        name_img = img.split(EXTENSION)[0]
        img = cv2.imread(img_dir + name_img + EXTENSION)
        height, width, channels = img.shape
        print('name_img:',name_img)
        print('height: ', height)
        print('width: ', width)
        print('channels: ', channels)
        img_crop = crop_img(img, crop_bondary)
        cv2.imwrite(img_save + name_img + EXTENSION , img_crop)

crop_bondary = [40, 180, 300, 420]
img_dir = '/home/yzi/Proposal/code/ice_detection/data/images/'
img_save = '/home/yzi/Proposal/code/ice_detection/data/crop_img/'
batch_crop_img(img_dir, img_save, crop_bondary)
# %% img to grey
import os
import cv2
import numpy as np

def batch_2GRAY_img(img_dir, img_save):
    EXTENSION = '.jpg'
    imgs_full = os.listdir(img_dir)
    print('Processing ' + str(len(imgs_full)) + ' number of images in the current folder')
    for img in imgs_full:
        # Read images using OpenCV and normalize into grayscale
        name_img = img.split(EXTENSION)[0]
        print('name_img:',name_img)
        img = cv2.imread(img_dir + name_img + EXTENSION)
        height, width, channels = img.shape
        print('name_img:',name_img)
        print('height: ', height)
        print('width: ', width)
        print('channels: ', channels)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_idx = int(name_img.split('imagesimg')[1]) - 280
        cv2.imwrite(img_save + 'img'+'_'+ str(new_idx)+ EXTENSION , gray)

img_dir = '/home/yzi/Proposal/code/ice_detection/data/crop_img/'
img_save = '/home/yzi/Proposal/code/ice_detection/data/gray_img/'
batch_2GRAY_img(img_dir, img_save)
# %% json to mask image size: (300, 420)
import json
path = '/home/yzi/Proposal/code/ice_detection/result/whole_road_region.json'
import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

from labelme.logger import logger


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

with open(path, "r",encoding="utf-8") as f:
    dj = json.load(f)
# dj['shapes'][0]Is for one label this time.
mask = shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][0]['points'], shape_type=None,line_width=1, point_size=1)
mask_img = mask.astype(np.int)#boolean to 0,Convert to 1

#I'm using anaconda
import matplotlib.pyplot as plt

plt.imshow(mask_img)
plt.show()
# %%
import cv2
cv2.imwrite('road_mask.jpg', mask_img*255)
# %% draft of training
import numpy as np
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
import  torch
from torch import nn
from torch.autograd import Variable

# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 数据预处理，标准化
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x
# %%
train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
# %%
from PIL import Image
import glob, os

size = 128, 128

for infile in glob.glob("/home/yzi/Proposal/code/ice_detection/data/normal/*.jpg"):
    file, ext = os.path.splitext(infile)
    print('infile: ', infile)
    im = Image.open(infile)
    print('im: ',im.size)
    # with Image.open(infile) as im:
    #     im.thumbnail(size)
    #     im.save(file + ".thumbnail", "JPEG")
