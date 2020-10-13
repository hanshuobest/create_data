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


import pdb
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


imgFolderName = None

@functools.lru_cache(maxsize=12)
def parese_json(fpath):
    with open(fpath, 'rb') as f:
        myjson = json.load(f)
        return myjson

def update_PolygonPoints(polygon_pts , top_left_pt):
    '''
    根据起始点更新多边形的每个顶点
    author:hanshuo
    :param polygon_pts: [[x1,y1] , [x2 , y2] ,... [xn , yn]]
    :param top_left_pt: [x,y]
    :return:
    '''
    num_poly = len(polygon_pts)
    for i in range(num_poly):
        polygon_pts[i][0] = int(polygon_pts[i][0]) + top_left_pt[0]
        polygon_pts[i][1] = int(polygon_pts[i][1]) + top_left_pt[1]

def isLineCross(pt1_1 , pt1_2 , pt2_1 , pt2_2):
    '''
    判断两条线是否相交
    :param pt1_1:
    :param pt1_2:
    :param pt2_1:
    :param pt2_2:
    :return:True相交
    '''
    ret = min(pt1_1[0] , pt1_2[0]) <= max(pt2_1[0] , pt2_2[0]) and min(pt2_1[0] , pt2_2[0]) <= max(pt1_1[0] , pt1_2[0]) and min(pt1_1[1] , pt1_2[1]) <= max(pt2_1[1] , pt2_2[1]) and min(pt2_1[1] , pt2_2[1]) <= max(pt1_1[1] , pt1_2[1])
    return ret

def isCross(p1 , p2 , p3):
    '''
    跨立实验
    :param p1:
    :param p2:
    :param p3:
    :return:
    '''
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1

def IsIntersec(p1,p2,p3,p4):
    '''
    判断两条直线是否相交
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :return:
    '''
    flag = False
    if isLineCross(p1 , p2 , p3 , p4):
        if(isCross(p1,p2,p3) * isCross(p1,p2,p4)<=0 and isCross(p3,p4,p1) * isCross(p3,p4,p2) <= 0):
            flag = True
        else:
            flag = False
    else:
        flag = False
    return flag

def getCrossPoint(p1 , p2 , q1 , q2):
    '''
    计算两条直线的交点
    :param pt1:
    :param pt2:
    :param pt3:
    :param pt4:
    :return:
    '''
    jiaodian = []
    if IsIntersec(p1 , p2 , q1 , q2):
        # 求交点
        left = (q2[0] - q1[0]) * (p1[1] - p2[1]) - (p2[0] - p1[0]) * (q1[1] - q2[1])
        right = (p1[1] - q1[1]) * (p2[0] - p1[0]) * (q2[0] - q1[0]) + q1[0] * (q2[1] - q1[1]) * (p2[0] - p1[0]) - p1[0] * (p2[1] - p1[1]) * (q2[0] - q1[0])

        if left == 0:
            return jiaodian
        x = int(float(right)/float(left))
        left = (p1[0] - p2[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q1[0] - q2[0])
        right = p2[1] * (p1[0] - p2[0]) * (q2[1] - q1[1]) + (q2[0] - p2[0]) * (q2[1] - q1[1]) * (p1[1] - p2[1]) - q2[1] * (q1[0] - q2[0]) * (p2[1] - p1[1])
        if left == 0:
            return jiaodian
        y = int(float(right)/float(left))

        jiaodian.append(x)
        jiaodian.append(y)


    return jiaodian

def isInPolygon_2(polygon_pts , pt):
    '''

    :param polygon_pts:
    :param pt:
    :return: 点在多边形内则返回True , 否则返回False
    '''
    num_pts = len(polygon_pts)
    if num_pts < 2:
        return False
    nCross = 0
    for i in range(num_pts):
        pt_A = polygon_pts[i]
        pt_B = polygon_pts[(i + 1) % num_pts]

        if pt_A[1] == pt_B[1]:
            continue
        if pt[1] < min(pt_A[1] , pt_B[1]):
            continue
        if pt[1] >= max(pt_A[1] , pt_B[1]):
            continue
        # 求交点x坐标
        x = float(pt[1] - pt_A[1]) * float(pt_B[0] - pt_A[0])/float(pt_B[1] - pt_A[1]) + pt_A[0]
        # 只统计单边交点
        if x > pt[0]:
            nCross += 1
    # 单边交点为偶数，则点在多边形外
    return nCross %2 == 1

def PointCmp(a , b , center):
    '''
    若点a大于点b，即点a在点b顺时针方向，返回True，否则返回False
    :param a:
    :param b:
    :param center:
    :return:
    '''
    if a[0] >= 0 and b[0] < 0:
        return True
    if a[0] == 0 and b[0] == 0:
        return a[1] > b[1]
    # 向量OA和向量OB的叉积
    det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])
    if det < 0:
        return True
    if det > 0:
        return False

    # 共线，以A、B距离center的距离判断大小
    d1 = (a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1])
    d2 = (b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1])

    return d1 > d2

def ClockwiseSortpoints(poly_pts):
    '''
    点集排序
    :param poly_pts:
    :return:
    '''
    # 计算重心
    center = []
    x = 0
    y = 0
    num = len(poly_pts)
    for i in range(num):
        x += poly_pts[i][0]
        y += poly_pts[i][1]

    center.append(int(x/num))
    center.append(int(y/num))

    # 排序
    for i in range(num - 1):
        for j in range(num - i - 1):
            if PointCmp(poly_pts[j] , poly_pts[j + 1] , center):
                tmp = poly_pts[j]
                poly_pts[j] = poly_pts[j + 1]
                poly_pts[j + 1] = tmp


def getPolygonCrossPoint(poly_pts1 , poly_pts2):
    '''
    计算两个多边形的交点
    :param poly_pts1:
    :param poly_pts2:
    :return:
    '''
    jiaodian = set()
    num1 = len(poly_pts1)
    num2 = len(poly_pts2)
    if num1 <3 or num2 < 3:
        return list(jiaodian)
    for i in range(num1):
        poly_next_idx = (i + 1) % num1
        for j in range(num2):
            poly2_next_idx = (j + 1) % num2
            pt = getCrossPoint(poly_pts1[i] , poly_pts1[poly_next_idx] , poly_pts2[j] , poly_pts2[poly2_next_idx])

            if pt:
                jiaodian.add(tuple(pt))
    jiaodian = list(jiaodian)

    # 如果没有交点
    if len(jiaodian) == 0:
        return jiaodian
    else:
        for j in jiaodian:
            j = list(j)
    return jiaodian

def updatePolygon(polygon_pts1 , polygon_pts2 , cross_pts):
    '''
    根据两个多边形相交，重新计算其重叠后的多边形
    :param polygon_pts1:
    :param polygon_pts2:
    :return:
    '''
    # random_seed = random.randint(0 , 1)
    # # 如果random_seed等于0 则poly1覆盖poly2，如果random_seed等于1，则poly2覆盖poly1

    new_poly1 = []
    new_poly2 = []
    new_poly2 = polygon_pts2
    for i in cross_pts:
        new_poly1.append(i)
    for j in polygon_pts1:
        if isInPolygon_2(polygon_pts2 , j):
            continue
        new_poly1.append(j)
    ClockwiseSortpoints(new_poly1)

    return new_poly1 , new_poly2


def calcIOU(x1, y1, w1, h1, x2, y2, w2, h2):
    '''

    :param x1:center_x
    :param y1:center_y
    :param w1:
    :param h1:
    :param x2:
    :param y2:
    :param w2:
    :param h2:
    :return:
    '''
    IOU = 0
    if ((abs(x1 - x2) < ((w1 + w2) / 2.0)) and (abs(y1 - y2) < ((h1 + h2) / 2.0))):
        left = max((x1 - (w1 / 2.0)), (x2 - (w2 / 2.0)))
        upper = max((y1 - (h1 / 2.0)), (y2 - (h2 / 2.0)))

        right = min((x1 + (w1 / 2.0)), (x2 + (w2 / 2.0)))
        bottom = min((y1 + (h1 / 2.0)), (y2 + (h2 / 2.0)))
        inter_w = abs(left - right)
        inter_h = abs(upper - bottom)
        inter_square = inter_w * inter_h
        union_square = (w1 * h1) + (w2 * h2) - inter_square
        IOU = inter_square / union_square * 1.0

    return IOU


def maxIou(x1, y1, w1, h1, x2, y2, w2, h2):
    IOU = 0
    if ((abs(x1 - x2) < ((w1 + w2) / 2.0)) and (abs(y1 - y2) < ((h1 + h2) / 2.0))):
        left = max((x1 - (w1 / 2.0)), (x2 - (w2 / 2.0)))
        upper = max((y1 - (h1 / 2.0)), (y2 - (h2 / 2.0)))
        right = min((x1 + (w1 / 2.0)), (x2 + (w2 / 2.0)))
        bottom = min((y1 + (h1 / 2.0)), (y2 + (h2 / 2.0)))
        inter_w = abs(left - right)
        inter_h = abs(upper - bottom)
        inter_square = inter_w * inter_h
        area1 = w1 * h1
        area2 = w2 * h2
        iou1 = float(inter_square) / area1
        iou2 = float(inter_square) / area2
        if iou1 > iou2:
            return iou1
        else:
            return iou2
    return IOU

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

def warfPoints(poly_pts , matrix_):
    '''
    对多边形点进行仿射变换
    :param poly_pts:
    :param matrix_:
    :return:
    '''
    # rotate_pts = []
    # for i in range(len(poly_pts)):
    #     tmp = np.append(poly_pts[i] , [1]).reshape((3 , 1))
    #     x_y = np.matmul(matrix_ , tmp)
    #     rotate_pts.append([x_y[0][0] , x_y[1][0]])
    # return rotate_pts
    
    
    array_poly_pts = np.array(poly_pts)
    array_one = np.array([1 for i in range(len(poly_pts))])
    array_one = np.expand_dims(array_one , 1)
    array_poly_pts = np.hstack((array_poly_pts, array_one))
    
    array_rotate_pts = array_poly_pts @ matrix_.T
    return array_rotate_pts.tolist()  
    

def imageZoom(img ,poly_pts ,  flag):
    '''
    缩放操作
    :param img:
    :param poly_pts:
    :param flag:
    :return:
    '''
    img_h , img_w = img.shape[:2]
    new_h = 0
    new_w = 0

    if flag == 1:
        # new_h = int(1.1 * img_h)
        # new_w = int(1.1 * img_w)
        
        random_val = random.randint(1 , 3)
        new_h = int(float(random_val * 0.1 + 1) * img_h)
        new_w = int(float(random_val * 0.1 + 1) * img_w)
    else:
        new_h = int(img_h/1.1)
        new_w = int(img_w/1.1)

    img2 = cv2.resize(img , (new_w , new_h))
    
    poly_pts = np.array(poly_pts)
    
    new_poly = poly_pts/np.array([img_w , img_h]) * np.array([new_w , new_h])
    new_poly = new_poly.astype(np.int)
    
    return img2 , new_poly.tolist()
    
    # new_poly_pts = []
    # for i in poly_pts:
    #     new_x = int(new_w * i[0] / float(img_w))
    #     new_y = int(new_h * i[1] / float(img_h))
    #     new_poly_pts.append([new_x, new_y])
        
    # import pdb; pdb.set_trace()    
    # return  img2 , new_poly_pts

def addPepperNoise(src):
    '''
    添加椒盐噪声
    :param src:
    :return:
    '''
    img_h , img_w = src.shape[:2]
    new_img = src.copy()
    NoiseNum = int(0.001 * img_h * img_w)
    for i in range(NoiseNum):
        randX = np.random.randint(0 , img_w - 1)
        randY = np.random.randint(0 , img_h - 1)
        if np.random.randint(0 , 2) == 0:
            new_img[randY , randX] = [0 , 0 , 0]
        else:
            new_img[randY, randX] = [255, 255, 255]

    return new_img

def getLineLength(p1 , p2):
    l = math.pow((p1[0] - p2[0]) , 2) + math.pow((p1[1] - p2[1]) , 2)
    l = math.sqrt(l)

    return l

def getAreaOfTraingle(p1 , p2 , p3):
    '''
    海伦公式，计算三角形面积
    :param p1:
    :param p2:
    :param p3:
    :return:
    '''
    area = 0
    p1p2 = getLineLength(p1 , p2)
    p2p3 = getLineLength(p2 , p3)
    p3p1 = getLineLength(p3 , p1)

    s = (p1p2 + p2p3 + p3p1) * 0.5

    # 海伦公式
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
    area = math.sqrt(area)
    return area



def getAreaPoly(points):
    area = 0
    num = len(points)
    if num < 3:
        raise Exception('error')

    p1 = points[0]
    for i in range(1 , num - 1):
        p2 = points[i]
        p3 = points[i + 1]

        vec_p1p2 = (p2[0] - p1[0] , p2[1] - p1[1])
        vec_p2p3 = (p3[0] - p2[0] , p3[1] - p2[1])

        sign = 0

        # 判断正负
        vecMult = vec_p1p2[0] * vec_p2p3[1] - vec_p1p2[1] * vec_p2p3[0]
        if vecMult > 0:
            sign = 1
        elif vecMult < 0:
            sign = -1
        triArea = getAreaOfTraingle(p1 , p2 , p3) * sign

        area += triArea
    return abs(area)

def process_sub_roi(json_lsts):
    assert len(json_lsts) != 0
    global roi_path
    for j_info in json_lst:
        json_info = parese_json(j_info)
        polygon_points = json_info["shapes"][0]["points"]
        label = json_info["shapes"][0]["label"]
        imagePath = os.path.join(os.getcwd(), "sku", os.path.basename(j_info)[:-4] + suffix)
        img = cv2.imread(imagePath)
        if type(img) == type(None):
            print ("打开图片失败！")
            continue
        img_h, img_w, _ = img.shape
        
        pts = np.array(polygon_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        for i in range(y, y + h - 1):
            for j in range(x, x + w - 1):
                
                is_flag = isInPolygon_2(polygon_points, [j, i])
                # print('flag:' , is_flag)
                if is_flag:
                    continue
                else:
                    img[i, j] = [0, 0, 0]
        
        imgROI = img[y + 1: y + h - 1, x + 1: x + w - 1, :]
        
        roi_polygon_points = []
        for pp in polygon_points:
            roi_polygon_points.append([pp[0] - x, pp[1] - y])
        
        roi_json = {}
        roi_json["points"] = roi_polygon_points
        
        roi_imagePath = roi_path + "/" + os.path.basename(j_info)[:-5] + ".jpg"
        cv2.imwrite(roi_imagePath, imgROI)
        roi_jsonPath = roi_path + "/" + os.path.basename(j_info)[:-5] + "-" + "roi.json"
        
        with open(roi_jsonPath, 'w') as json_file:
            json.dump(roi_json, json_file, ensure_ascii=False)
            

def process_sub(background_lst , roi_lst):
    print('------start-------')
    global imgFolderName
    rotate_angles = [random.randint(-15 , 15) for i in range(5)]
    
    roi_num = len(roi_lst)
    np.random.seed(0)
    import pdb; pdb.set_trace()
    for i in tqdm(range(len(background_lst))):
        img = cv2.imread(background_lst[i])
        img_h, img_w, _ = img.shape
    
        pt1_x = np.random.randint(0, img_w)
        pt1_y = np.random.randint(0, img_h)

        roi_index1 = np.random.choice(range(roi_num), int(roi_num * 0.01), replace=False)
        if roi_index1.size == 0:
            roi_index1 = np.array([0])
    
        for j in roi_index1:  # 第一次遍历roi
            result = img.copy()
            roi_img_1 = cv2.imread(roi_lst[j])
            if type(roi_img_1) == type(None):
                print ('读取roi图片失败')
                break
            # 获取roi的多边形坐标
            poly_roi_img1 = parese_json(roi_lst[j][:-4] + "-roi.json")['points']
            label_1 = os.path.basename(roi_lst[j]).split('-')[0]
        
            # 图片缩放
            if np.random.randint(0, 2):
                roi_img_1, poly_roi_img1 = imageZoom(roi_img_1, poly_roi_img1, random.randint(0, 1))
                
                
            rotate_angles_index1 = np.random.choice(rotate_angles, 1, replace=False)
            # 遍历旋转角度
            for j_angle in rotate_angles_index1:
                result2 = result.copy()
                rotate_roi_img_1, rotate_matrix_1 = rotate_image(roi_img_1, j_angle)
                rotate_roi_h_1, rotate_roi_w_1, _ = rotate_roi_img_1.shape
                
                if rotate_roi_h_1 > img_h or rotate_roi_w_1 > img_w:
                    print("旋转后的roi尺寸:({},{}) ， 大于背景图尺寸({},{})".format(rotate_roi_h_1 , rotate_roi_w_1 , img_h, img_w))
                    continue
            
                if (pt1_x + rotate_roi_w_1) > img_w:
                    pt1_x = img_w - rotate_roi_w_1
                if (pt1_y + rotate_roi_h_1) > img_h:
                    pt1_y = img_h - rotate_roi_h_1
                poly_roi_img1 = np.array(poly_roi_img1, dtype=np.int32)
                # 根据仿射变换计算旋转后的轮廓点
                rotate_poly_roi_img1 = warfPoints(poly_roi_img1, rotate_matrix_1)
                update_PolygonPoints(rotate_poly_roi_img1, [pt1_x, pt1_y])
                
                rotate_roi_1 = result2[pt1_y: pt1_y + rotate_roi_h_1, pt1_x: pt1_x + rotate_roi_w_1, :]
                
                for i_1 in range(rotate_roi_h_1):
                    for j_1 in range(rotate_roi_w_1):
                        
                        if int(np.sum(rotate_roi_img_1[i_1][j_1])) <= 20:
                            continue
                        else:
                            rotate_roi_1[i_1][j_1] = rotate_roi_img_1[i_1][j_1]
            
                result2[pt1_y: pt1_y + rotate_roi_h_1, pt1_x: pt1_x + rotate_roi_w_1, :] = rotate_roi_1
                
                
                rotate_poly_roi_img1 = np.array(rotate_poly_roi_img1 , dtype=np.int32)
                rotate_poly_roi_img1 = rotate_poly_roi_img1.reshape((-1 , 1 , 2))
                x1, y1, w1, h1 = cv2.boundingRect(rotate_poly_roi_img1)
                
                
                imgFileName = os.path.basename(background_lst[i])[:-4] + "_" + os.path.basename(roi_lst[j])[:-4] + "_" + str(j_angle) + ".jpg"
                
                imagePath = os.path.join(imgFolderName, imgFileName)
                writer = PascalVocWriter(imgFolderName, imgFileName, (img_h, img_w, 3), localImgPath=imagePath,
                                                 usrname="auto")
                writer.verified = True
                writer.addBndBox(x1, y1, x1 + w1, y1 + h1, label_1, 0)
 
                writer.save(targetFile=imagePath[:-4] + XML_EXT)
                    
                if np.random.randint(1, 11) == 10:
                    result4 = addPepperNoise(result2)
                if np.random.randint(1, 6) == 3:
                    result4 = cv2.GaussianBlur(result2, (5, 5), 0)
                    
                cv2.imwrite(imagePath, result2)

if __name__ == '__main__':
    # 生成模板图像
    print('开始生成ROI模板')
    json_lst = glob.glob(os.path.join(os.getcwd(), "sku") + "/*.json")
    img_lst = glob.glob(os.path.join(os.getcwd(), "sku") + "/*.png") + glob.glob(os.path.join(os.getcwd() , "sku") + "/*.jpg")

    suffix = None
    print(len(json_lst))
    print(len(img_lst))
    assert len(json_lst) == len(img_lst)

    roi_path = "sku/roi"
    if os.path.exists(roi_path):
        shutil.rmtree(roi_path)
    else:
        os.mkdir(roi_path)
    if os.path.isdir(roi_path):
        pass
    else:
        os.mkdir(roi_path)

    start_time = time.time()
    for j_info in json_lst:
        json_info = parese_json(j_info)
        polygon_points = json_info["shapes"][0]["points"]
        label = json_info["shapes"][0]["label"]
        if os.path.exists(j_info[:-5] + ".jpg"):
            suffix = "jpg"
        else:
            suffix = "png"
        imagePath = os.path.join(os.getcwd(), "sku" , os.path.basename(j_info)[:-4] + suffix)
        img = cv2.imread(imagePath)
        if type(img) == type(None):
            print ("打开图片失败！")
            continue
        img_h, img_w, _ = img.shape

        pts = np.array(polygon_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        for i in range(y, y + h - 1):
            for j in range(x, x + w - 1):

                is_flag = isInPolygon_2(polygon_points, [j, i])
                if is_flag:
                    continue
                else:
                    img[i, j] = [0, 0, 0]

        imgROI = img[y + 1: y + h - 1, x + 1: x + w - 1, :]

        roi_polygon_points = []
        for pp in polygon_points:
            roi_polygon_points.append([pp[0] - x, pp[1] - y])

        roi_json = {}
        roi_json["points"] = roi_polygon_points

        roi_imagePath = roi_path + "/" + os.path.basename(j_info)[:-5] + ".jpg"
        cv2.imwrite(roi_imagePath, imgROI)
        roi_jsonPath = roi_path + "/" + os.path.basename(j_info)[:-5] + "-" + "roi.json"

        with open(roi_jsonPath, 'w') as json_file:
            json.dump(roi_json, json_file, ensure_ascii=False)
    
    
    cost_time = time.time() - start_time
    print('cost time:' , cost_time)
    print('模板ROI生成完毕！')

    # background_imgs = glob.glob(os.path.join(os.getcwd() , "sku" , "background") + "/*.png")
    background_imgs = list(list_images(os.path.join(os.getcwd() , "sku" , "background")))
    background_num = len(background_imgs)
    if background_num == 0:
        print ('没有背景图片')
        sys.exit(1)

    roi_imgs = glob.glob(os.path.join(os.getcwd() , "sku" , "roi") + "/*.jpg")
    roi_num = len(roi_imgs)

    generate_dir = os.path.join(os.getcwd() , "sku" , "generate")
    if os.path.exists(generate_dir):
        shutil.rmtree(generate_dir)
    else:
        os.mkdir(generate_dir)

    if os.path.isdir(generate_dir):
        pass
    else:
        os.mkdir(generate_dir)

    imgFolderName = generate_dir

    process_lst = []
    start_time = time.time()
    
    process_sub(background_imgs , roi_imgs)
    
    # classes_count = 1
    # for i in range(classes_count):
    #     ps = multiprocessing.Process(target=process_sub , args=(background_imgs[i * classes_count:(i + 1) * classes_count] , roi_imgs))
    #     ps.daemon = True
    #     process_lst.append(ps)
    # for i in range(classes_count):
    #     process_lst[i].start()

    # for  i in range(classes_count):
    #     process_lst[i].join()


    cost_time = time.time() - start_time
    print('cost time:' , cost_time)
    
    
    
    








