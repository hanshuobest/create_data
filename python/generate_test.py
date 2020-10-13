#coding:utf-8

import cv2
import os
import glob
import math
import numpy as np
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import XML_EXT
from PIL import Image , ImageDraw
import random
from collections import namedtuple
import logging
import argparse
import pika

BBOX = namedtuple("BBOX" , "xmin ymin xmax ymax")

# 计算两个bbox的最大交并比
def maxIOU(x1 , y1 , w1 , h1 , x2 , y2 , w2 , h2):
    W = min(x1 + w1 , x2 + w2) - max(x1 , x2)
    H = min(y1 + h1 , y2 + h2) - max(y1 , y2)

    if W <= 0 or H <= 0:
        return 0
    SA = w1 * h1
    SB = w2 * h2
    cross_area = H * W

    return max(float(cross_area)/float(SA) , float(cross_area)/float(SB))

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask ， 经过二值化后
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def rotate_image(image , angle):
    '''
    图像旋转
    :param image:
    :param angle:
    :return: 返回旋转后的图像以及旋转矩阵
    '''
    (h , w) = image.shape[:2]
    (cx , cy) = (int(0.5 * w) , int(0.5 * h))
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])

    # 计算新图像的bounding
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    return cv2.warpAffine(image , M , (nW , nH)) , M

if __name__ == '__main__':
    txt_lsts = []
    txt_lsts.append("eefb5090_0_8.txt")

    # 保存手的信息文件
    shape_dir = "/Users/han/generate_data/shapes"
    mask_foreground_lsts = glob.glob(os.path.join(shape_dir , "*.png"))
    assert len(mask_foreground_lsts) % 2 == 0

    # 存储手的信息
    img_infos = {}
    # 前景图片列表
    foreground_lsts = []
    for i in mask_foreground_lsts:
        if "foreground" in i:
            foreground_img = cv2.imread(i)
            foreground_img = cv2.cvtColor(foreground_img , cv2.COLOR_BGR2RGB)
            mask_img = cv2.imread(i.replace("foreground" , "mask") , 0)
            foreground_lsts.append(i)
            img_infos[i] = []
            img_infos[i].append(foreground_img)
            img_infos[i].append(mask_img)

    num_foreground = len(foreground_lsts)


    for i in txt_lsts:
        img_file = i.replace("txt" , "jpg")

        if not os.path.exists(img_file):
            continue
        img = cv2.imread(img_file)
        # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img_h , img_w = img.shape[:2]

        random_hand_index = np.random.choice(np.arange(num_foreground) , 1)
        mask_shape_img = img_infos[foreground_lsts[random_hand_index[0]]][1]
        foreground_shape_img = img_infos[foreground_lsts[random_hand_index[0]]][0]

        rotate_shape_angle = random.randint(-30, 30)
        mask_shape_img , _ = rotate_image(mask_shape_img , rotate_shape_angle)
        foreground_shape_img , _ = rotate_image(foreground_shape_img , rotate_shape_angle)

        _ , mask_shape_binary = cv2.threshold(mask_shape_img , 0 , 255 , cv2.THRESH_BINARY)
        r_xmin , r_xmax , r_ymin , r_ymax = get_annotation_from_mask(mask_shape_binary)


        mask_shape_img = mask_shape_img[r_ymin:r_ymax , r_xmin:r_xmax]
        foreground_shape_img = foreground_shape_img[r_ymin:r_ymax , r_xmin:r_xmax]

        mask_shape_img = Image.fromarray(mask_shape_img)
        foreground_shape_img = Image.fromarray(foreground_shape_img)

        # img = Image.fromarray(np.uint8(img))
        mask_w = mask_shape_img.width
        mask_h = mask_shape_img.height

        class_dict = {}

        try:
            with open(i) as fp:
                for line in fp.readlines():
                    infos = line.split("\t")

                    x = infos[1].strip()
                    y = infos[2].strip()
                    z = infos[3].strip()

                    label = infos[0].strip()

                    if class_dict.has_key(label):
                        class_dict[label].append((int(x), img_h - int(y), float(z)))
                    else:
                        class_dict[label] = []
        except:
            continue


        boundingBox = []
        for key , val  in class_dict.items():

            xmin = np.Inf
            xmax = -np.Inf
            ymin = np.Inf
            ymax = -np.Inf

            for j in val:
                if xmin > j[0]:
                    xmin = j[0]
                if xmax < j[0]:
                    xmax = j[0]
                if ymin > j[1]:
                    ymin = j[1]
                if ymax < j[1]:
                    ymax = j[1]

            xmin = max(0, xmin)
            xmax = min(xmax, img_w)
            ymin = max(0, ymin)
            ymax = min(ymax, img_h)

            print('key:' , key)
            print('xmin:' , xmin)
            print('xmax:' , xmax)
            print('ymin:' , ymin)
            print('ymax:' , ymax)

            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(img, str(key), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255 , 0 , 0), 2)
        cv2.imshow("img" , img)
        cv2.waitKey(0)

    print("finished!")




