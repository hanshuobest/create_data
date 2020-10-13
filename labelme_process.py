#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :labelme_process.py
@brief       :利用labelme进行处理生成掩膜文件
@time        :2020/10/13 16:18:54
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''

# coding:utf-8

import os
import glob
import numpy as np
from PIL import Image
import shutil
import argparse


def is_dir_exist(model_path):
    if os.path.exists(model_path):
        pass
    else:
        os.mkdir(model_path)


def movefile(source, dest):
    try:
        shutil.move(source, dest)
    except shutil.Error:
        if os.path.isdir(source):
            shutil.rmtree(source)
        else:
            os.remove(source)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
                        help="sku directory", default="sku")
    args = parser.parse_args()
    args = vars(args)

    sku_logo = "sku_logo"
    save_path = os.path.join(args['directory'], sku_logo)
    is_dir_exist(save_path)

    # 处理labelme生成的json文件
    json_lsts = glob.glob(args['directory'] + "/*.json")
    for i in json_lsts:
        str_commod = "labelme_json_to_dataset" + " " + i
        os.system(str_commod)

        result_path = i[:-5] + "_json" 
        image_name = i.replace("jpg" , "json")
        if os.path.exists(image_name):
            movefile(image_name , result_path)
        movefile(i , result_path)
        movefile(result_path , save_path)

    print('finished!')
