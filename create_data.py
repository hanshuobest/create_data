#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :create_data.py
@brief       :
@time        :2020/10/12 17:49:29
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''


from os import write
import pdb
from random import randint
import cv2
import os
import glob
import sys
import numpy as np
import json
import random
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import XML_EXT
import shutil
from tqdm import tqdm
import math
import multiprocessing
import time
import traceback
from imutils.paths import list_images
import functools
import argparse
import configparser
from os.path import join, exists, basename
from add_logo import add_logo
from copy import deepcopy


imgFolderName = None


def resize_img_keep_ratio(cv_img, target_size=(720, 1280)):
    old_size = cv_img.shape[0:2]

    ratio = min(float(target_size[i])/(old_size[i])
                for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv2.resize(cv_img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h//2, pad_h-(pad_h//2)
    left, right = pad_w//2, pad_w - (pad_w//2)
    img_new = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new, pad_w, pad_h


def addPepperNoise(src):
    '''
    添加椒盐噪声
    :param src:
    :return:
    '''
    img_h, img_w = src.shape[:2]
    new_img = src.copy()
    NoiseNum = int(0.001 * img_h * img_w)
    for i in range(NoiseNum):
        randX = np.random.randint(0, img_w - 1)
        randY = np.random.randint(0, img_h - 1)
        if np.random.randint(0, 2) == 0:
            new_img[randY, randX] = [0, 0, 0]
        else:
            new_img[randY, randX] = [255, 255, 255]

    return new_img


def read_labels(label_file):
    with open(label_file, 'r') as fp:
        all_lines = fp.readlines()
        if len(all_lines) == 2:
            return all_lines[1]
        else:
            return None


def process_sub(background_lst, roi_lst):
    print('------start-------')
    global imgFolderName

    roi_num = len(roi_lst)
    np.random.seed(0)

    for i in tqdm(range(len(background_lst))):
        bk_img = cv2.imread(background_lst[i])
        if type(bk_img) == type(None):
            continue
        bk_height, bk_width = bk_img.shape[:2]

        for j in roi_lst:
            logo_src_img_name = join(j, "img.png")
            mask_img_name = join(j, "label.png")
            label_name = join(j, "label_names.txt")

            if not exists(logo_src_img_name) or not exists(mask_img_name):
                print("mask file: {} or logo file: {} do not exist".format(
                    logo_src_img_name, mask_img_name))
                continue

            logo_src_img_cv = cv2.imread(logo_src_img_name)
            mask_img_cv = cv2.imread(mask_img_name, 0)
            label_name = read_labels(label_name)

            result_img, bndbox = add_logo(
                deepcopy(bk_img), logo_src_img_cv, mask_img_cv)
            resize_result_img, pad_w, pad_h = resize_img_keep_ratio(
                result_img, target_size=(720, 1280))
            pad_w = pad_w >> 1
            pad_h = pad_h >> 1

            h_scale = 720 / bk_height
            w_scale = 1280 / bk_width
            scale = min(h_scale, w_scale)
            bndbox = (int(scale * bndbox[0]) + pad_w, int(scale * bndbox[1]) + pad_h, int(
                scale * bndbox[2]) + pad_w, int(scale * bndbox[3]) + pad_h)

            if bndbox[2] - bndbox[0] <= 20 or bndbox[3] - bndbox[1] <= 20:
                continue

            imgFileName = basename(background_lst[i])[
                :-4] + "_" + basename(j) + ".jpg"
            write_xml(imgFileName, imgFolderName,
                      result_img.shape, bndbox, label_name)

            if randint(0, 10) == 8:
                resize_result_img = cv2.GaussianBlur(
                    resize_result_img, (5, 5), 0)

            cv2.imwrite(join(imgFolderName, imgFileName), resize_result_img)


def write_xml(image_file_name, img_folder_name, img_size, bndbox, label):
    imagePath = join(img_folder_name, image_file_name)
    writer = PascalVocWriter(imgFolderName, image_file_name, (
        img_size[0], img_size[1], 3), localImgPath=imagePath, usrname="auto")
    writer.verified = True
    writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, 0)
    writer.save(targetFile=imagePath[:-4] + XML_EXT)


def is_dir_exist(model_path):
    if os.path.exists(model_path):
        pass
    else:
        os.mkdir(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", help="sku directory", default="sku")
    args = vars(parser.parse_args())

    sku_dir_lst = os.listdir(join(args['input'], "sku_logo"))
    sku_dir_lst = list(
        map(lambda x: join(args['input'], "sku_logo", x), sku_dir_lst))

    background_imgs = list(list_images(join(args['input'], "background")))
    background_num = len(background_imgs)
    if background_num == 0:
        print('没有背景图片')
        sys.exit(1)

    generate_dir = os.path.join(args['input'], "generate")
    is_dir_exist(generate_dir)
    imgFolderName = generate_dir

    process_lst = []
    start_time = time.time()

    # process_sub(background_imgs, sku_dir_lst)

    process_count = multiprocessing.cpu_count()
    per_sku_number = int(len(sku_dir_lst)/ process_count)
    for i in range(process_count):
        if i * per_sku_number < len(sku_dir_lst):
            ps = multiprocessing.Process(target=process_sub , args=(background_imgs , sku_dir_lst[i * per_sku_number : (i + 1) * per_sku_number]))
        else:
            ps = multiprocessing.Process(target=process_sub , args=(background_imgs , sku_dir_lst[i * per_sku_number : ]))
        ps.daemon = True
        process_lst.append(ps)
    for i in range(process_count):
        process_lst[i].start()

    for  i in range(process_count):
        process_lst[i].join()
        
    print("cost time: {}".format(time.time() - start_time))
