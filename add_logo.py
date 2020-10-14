#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :add_logo.py
@brief       :往图片添加logo
@time        :2020/10/13 16:43:00
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''

import cv2 as cv
import os
import numpy as np
from imutils.paths import list_images
from random import randint
import sys


def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations
    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask_img = cv.imread(mask_file)
        _, mask = cv.threshold(mask_img, 10, 255, cv.THRESH_BINARY)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        return -1, -1, -1, -1


def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations
    Args:
        mask(NumPy Array): Array with the mask
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


def add_logo(bk_img_cv, logo_img_cv, mask_img_cv, start_x=0, start_y=0):
    assert logo_img_cv.shape != mask_img_cv.shape, print("size not same")
    bk_height , bk_width = bk_img_cv.shape[:2]
    _, mask_img = cv.threshold(mask_img_cv, 10, 255, cv.THRESH_BINARY)
    xmin, xmax, ymin, ymax = get_annotation_from_mask(mask_img)
    logo_height, logo_width = ymax - ymin, xmax - xmin
    
    mask_img = mask_img[ymin: ymax, xmin: xmax]
    logo_img_cv = logo_img_cv[ymin: ymax, xmin: xmax]
    
    scale = 1.0
    if logo_height * 3 >= bk_height or logo_width * 3 >= bk_width:
        scale = min(bk_height/(3 * logo_height), bk_width/(3 * logo_width))
    print("scale: ", scale)    
    print("src logo size: {} , {}".format(logo_height, logo_width))
    target_logo_size = (int(logo_height * scale), int(logo_width * scale))
    print("target log size: " , target_logo_size)
    
    mask_img = cv.resize(mask_img, (target_logo_size[1] , target_logo_size[0]))
    logo_img_cv = cv.resize(logo_img_cv, (target_logo_size[1] , target_logo_size[0]))
    roi_img = bk_img_cv[start_y:target_logo_size[0] + start_y, start_x:target_logo_size[1] + start_x]

    mask_img_inv = cv.bitwise_not(mask_img)
    img1_bg = cv.bitwise_and(roi_img, roi_img, mask=mask_img_inv)
    img2_fg = cv.bitwise_and(logo_img_cv, logo_img_cv, mask=mask_img)

    dst = cv.add(img1_bg, img2_fg)
    src_img[start_y:target_logo_size[0] + start_y , start_x:target_logo_size[1] + start_x] = dst
    return src_img , (int(scale * xmin), int(scale * ymin), int(scale * xmax) , int(scale * ymax))


def resize_img_keep_ratio(cv_img, target_size=(720, 1280)):
    old_size = cv_img.shape[0:2]

    ratio = min(float(target_size[i])/(old_size[i])
                for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv.resize(cv_img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h//2, pad_h-(pad_h//2)
    left, right = pad_w//2, pad_w - (pad_w//2)
    img_new = cv.copyMakeBorder(
        img, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


if __name__ == '__main__':
    path_dir = "sku/sku_logo/1_20201011220933_json"

    logo_file = os.path.join(path_dir, "img.png")
    mask_file = os.path.join(path_dir, "label.png")

    back_ground_dir = "sku/background"
    back_image_lst = list(list_images(back_ground_dir))
    if len(back_image_lst) == 0:
        print("没有背景图")
        sys.exit(0)

    back_image_name = back_image_lst[randint(0, len(back_image_lst) - 1)]
    logo_img = cv.imread(logo_file)
    mask_img = cv.imread(mask_file, 0)
    src_img = cv.imread(back_image_name)
    

    result_img , bndbox = add_logo(src_img, logo_img, mask_img , start_x = 100 , start_y = 100)
    result_img = resize_img_keep_ratio(result_img)
    cv.imwrite("result.jpg", result_img)
