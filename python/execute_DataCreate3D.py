#coding:utf-8

import os
import sys


if __name__ == '__main__':
    os.system("rm *.jpg")
    model_root_dir = "../models/models.txt"
    with open(model_root_dir , "r") as fp:
        all_lines = fp.readlines()

    for i in range(len(all_lines) * len(all_lines)):
        os.system("../DataCreate3D")

    os.system("python test_Ann_JP.py")
    os.system("rm ../*.txt")

    print(len(all_lines))
