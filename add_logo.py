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


def seamless_add_logo(cv_logo_img, cv_src_img):
    gray = cv.cvtColor(cv_logo_img, cv.COLOR_BGR2GRAY)

    _, mask_gray = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    # import pdb; pdb.set_trace()
    cv.imwrite("binary.jpg", mask_gray)
    # cv.imshow("binary" , binary_img)
    # cv.waitKey(0)

    logo_shape = cv_logo_img.shape[:2]

    mask = 255 * np.ones((logo_shape[0], logo_shape[1]), cv_logo_img.dtype)
    # import pdb; pdb.set_trace()

    output = cv.seamlessClone(cv_logo_img, cv_src_img,
                              mask, (500, 500), cv.NORMAL_CLONE)
    # cv.imshow("seamless", output)
    cv.imwrite("save.jpg", output)

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations
    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask_img = cv.imread(mask_file)
        _ , mask = cv.threshold(mask_img, 10, 255, cv.THRESH_BINARY)
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



def add_logo(bk_img_cv , logo_img_cv , mask_img_cv):
    assert logo_img_cv.shape != mask_img_cv.shape , print("size not same")
    _ , mask_img = cv.threshold(mask_img_cv, 10, 255, cv.THRESH_BINARY)
    
    xmin , xmax , ymin , ymax = get_annotation_from_mask(mask_img)
    logo_h , logo_w = ymax - ymin , xmax - xmin
    
    mask_img = mask_img[ymin : ymax , xmin : xmax]
    logo_img_cv = logo_img_cv[ymin : ymax , xmin : xmax]
    
    roi_img = bk_img_cv[:logo_h , :logo_w]
    
    mask_img_inv = cv.bitwise_not(mask_img)
    
    img1_bg = cv.bitwise_and(roi_img, roi_img , mask= mask_img_inv)
    import pdb; pdb.set_trace()
    img2_fg = cv.bitwise_and(logo_img_cv , logo_img_cv , mask= mask_img)
    
    dst = cv.add(img1_bg, img2_fg)
    src_img[:logo_h , :logo_w] = dst
    return src_img
    


if __name__ == '__main__':  
    path_dir = "sku/sku_logo/1_20201011220933_json"
    
    logo_file = os.path.join(path_dir, "img.png")
    mask_file = os.path.join(path_dir, "label.png")
    logo_img = cv.imread(logo_file)
    mask_img = cv.imread(mask_file , 0)
    src_img = cv.imread(
        "/home/han/project/git_project/sric_create_data/sku/background/b_living_20201010103321.jpg")
    
    result_img = add_logo(src_img, logo_img , mask_img)
    cv.imshow("result" , result_img)
    cv.waitKey(0)   

