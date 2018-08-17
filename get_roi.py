#coding:utf-8
import cv2
import json
import os
import numpy as np
import glob
import shutil
import sys
import math

def parese_json(fpath):
    with open(fpath, 'rb') as f:
        myjson = json.load(f)
        # imagePath = myjson["imagePath"]
        # print(imagePath)
        # print('shapes:' , myjson["shapes"])
        # print('points:' , myjson["shapes"][0]['points'])
        # print('label:'  , myjson["shapes"][0]['label'])
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

def getGravity(pts):
    '''
    计算多边形重心
    :param pts:
    :return:
    '''
    area = 0
    center = []
    num = len(pts)
    x = 0
    y = 0
    for i in range(num - 1):
        area += (pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]) * 0.5
        x += (pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]) * (pts[i][0] + pts[i + 1][0])
        y += (pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]) * (pts[i][1] + pts[i + 1][1])

    area += (pts[num - 1][0] * pts[0][1] - pts[0][0] * pts[num - 1][1]) * 0.5
    x += (pts[num - 1][0] * pts[0][1] - pts[0][0] * pts[num - 1][1]) * (pts[num - 1][0] + pts[0][0])
    y += (pts[num - 1][0] * pts[0][1] - pts[0][0] * pts[num - 1][1]) * (pts[num - 1][1] + pts[0][1])

    x /= 6 * area
    y /= 6 * area

    center.append(x)
    center.append(y)
    return center

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
    new_polys = poly_pts

    # 排序
    for i in range(num - 1):
        for j in range(num - i - 1):
            if PointCmp(new_polys[j] , new_polys[j + 1] , center):
                tmp = new_polys[j]
                new_polys[j] = new_polys[j + 1]
                new_polys[j + 1] = tmp
    return new_polys


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
    rotate_pts = []
    for i in range(len(poly_pts)):
        tmp = np.append(poly_pts[i] , [1]).reshape((3 , 1))
        x_y = np.matmul(matrix_ , tmp)
        rotate_pts.append([x_y[0][0] , x_y[1][0]])
    return rotate_pts

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
        new_h = int(1.1 * img_h)
        new_w = int(1.1 * img_w)
    else:
        new_h = int(img_h/1.1)
        new_w = int(img_w/1.1)

    img2 = cv2.resize(img , (new_w , new_h))
    new_poly_pts = []
    for i in poly_pts:
        new_x = int(new_w * i[0] / float(img_w))
        new_y = int(new_h * i[1] / float(img_h))
        new_poly_pts.append([new_x, new_y])
    return  img2 , new_poly_pts

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

def optimizerPolygon(jiaodian , pts1):
    '''
    根据交点优化pts1组成的多边形
    :param jiaodian:
    :param pts1:
    :return:
    '''

    assert len(jiaodian) == 4
    indexs = []
    for i in jiaodian:
        index = pts1.index(list(i))
        indexs.append(index)
    indexs.sort()
    print('indexs:' , indexs)

    pt_lsts = [[] for i in range(4)]
    pt_lsts[1] = pts1[indexs[0]:indexs[1] + 1]

    pt_lsts[2] = pts1[indexs[1]:indexs[2] + 1]
    pt_lsts[3] = pts1[indexs[2]:indexs[3] + 1]

    other_1 = pts1[:indexs[0] + 1]
    other_2 = pts1[indexs[3]:]

    other_set = set()
    num1 = len(other_1)
    num2 = len(other_2)
    if num1 == 0 and num2 != 0:
        other_set = list(other_2)
    elif num2 == 0 and num1 != 0:
        other_set = list(other_1)
    elif num1 != 0 and num2 != 0:
        for i in range(num1):
            other_set.add(tuple(other_1[i]))
        for i in range(num2):
            other_set.add(tuple(other_2[i]))
    pt_lsts[0] = list(other_set)

    total_area = getAreaPoly(pts1)
    for i in pt_lsts:
        if len(i) < 3:
            continue
        area = getAreaPoly(i)
        ratio = float(area) / total_area
        if ratio < 0.33:
            continue
        else:
            pts1 = i
    return pts1



if __name__ == '__main__':
    # json_lst = glob.glob(os.path.join(os.getcwd() , "sku") + "/*.json")
    # img_lst =  glob.glob(os.path.join(os.getcwd() , "sku") + "/*.png")
    #
    # assert len(json_lst) == len(img_lst)
    #
    # roi_path = "sku/roi"
    # if os.path.exists(roi_path):
    #     shutil.rmtree(roi_path)
    # else:
    #     os.mkdir(roi_path)
    # if os.path.isdir(roi_path):
    #     pass
    # else:
    #     os.mkdir(roi_path)
    #
    # for j_info in json_lst:
    #     json_info = parese_json(j_info)
    #     polygon_points = json_info["shapes"][0]["points"]
    #     label = json_info["shapes"][0]["label"]
    #     imagePath = os.path.join(os.getcwd() , "sku") + "/" + os.path.basename(json_info['imagePath'])
    #
    #     img = cv2.imread(imagePath)
    #     if type(img) == type(None):
    #         print "打开图片失败！"
    #         break
    #     img_h , img_w , _ = img.shape
    #     pts = np.array(polygon_points , dtype=np.int32)
    #     x , y , w , h = cv2.boundingRect(pts)
    #     # cv2.rectangle(img , (x , y) , (x + w , y + h) , (255 , 0 , 0) , thickness=2)
    #     for i in range(y , y + h):
    #         for j in range(x , x + w):
    #
    #             is_flag = isInPolygon_2(polygon_points , [j , i])
    #             # print('flag:' , is_flag)
    #             if is_flag:
    #                 continue
    #             else:
    #                 img[i , j] = [0 , 0 , 0]
    #
    #     imgROI = img[y : y + h , x : x + w , :]
    #     roi_polygon_points = []
    #     for pp in polygon_points:
    #         roi_polygon_points.append([pp[0] - x , pp[1] - y])
    #
    #     roi_json = {}
    #     roi_json["points"] = roi_polygon_points
    #
    #     roi_imagePath = roi_path + "/" + os.path.basename(json_info['imagePath'])[:-4] + ".png"
    #     cv2.imwrite(roi_imagePath , imgROI)
    #     roi_jsonPath = roi_path + "/" + os.path.basename(json_info['imagePath'])[:-4] + "-" + "roi.json"
    #     with open(roi_jsonPath , 'w') as json_file:
    #         json.dump(roi_json , json_file , ensure_ascii=False)
    #
    # cv2.waitKey(0)


    # roi = cv2.imread("/Users/han/Downloads/bn_exp/sku/roi/37-0.png")
    # if type(roi) == type(None):
    #     print '打开图片失败'
    #
    # json_info = parese_json("/Users/han/Downloads/bn_exp/sku/roi/37-0-roi.json")
    # img_h , img_w = roi.shape[:2]
    # pts = np.array(json_info["points"] , dtype=np.int32)
    # pts = pts.reshape((-1 , 1 , 2))
    #
    # new_img = addPepperNoise(roi)
    # copy_img = new_img.copy()
    # copy_img = cv2.GaussianBlur(copy_img , (5 , 5) , 0)
    #
    # cv2.imshow("copy" , copy_img)
    # cv2.imshow("new" , new_img)
    # cv2.waitKey(0)




    # img = cv2.imread("/Users/han/Downloads/bn_exp/sku/generate/10-37-0-120-61-0-150.png")
    # if type(img) == type(None):
    #     sys.exit(1)
    #
    # json_points1 = parese_json("/Users/han/Downloads/bn_exp/sku/generate/10-37-0-120.json")['points']
    # json_points2 = parese_json("/Users/han/Downloads/bn_exp/sku/generate/10-37-0-120-61-0-150.json")['points']
    #
    # print('json_points1:' , json_points1)
    # print('json_points2:' , json_points2)
    #
    # json_points1 = np.array(json_points1 , dtype=np.int32)
    # json_points2 = np.array(json_points2 , dtype=np.int32)
    # json_points1.reshape((-1 , 1 , 2))
    # json_points2.reshape((-1 , 1 , 2))
    #
    # cv2.polylines(img , [json_points1] , True , (0 , 0 , 255))
    # cv2.polylines(img , [json_points2] , True , (0 , 255 , 0))
    #
    # x1 , y1 , w1 , h1 = cv2.boundingRect(json_points1)
    # x2 , y2 , w2 , h2 = cv2.boundingRect(json_points2)
    #
    # cv2.rectangle(img , (x1 , y1) , (x1 + w1 , y1 + h1) , (255 , 0 , 0))
    # cv2.rectangle(img , (x2 , y2) , (x2 + w2 , y2 + h2) , (255 , 0 , 0))
    # rect1 = cv2.minAreaRect(json_points1)
    # print('rect1:' , rect1)
    # box1  = cv2.boxPoints(rect1)
    # box1 = np.int0(box1)
    # print(box1)
    # cv2.drawContours(img , [box1] , 0 , (255 , 0 , 0) , 3)
    #
    # rect2 = cv2.minAreaRect(json_points2)
    # box2 = cv2.boxPoints(rect2)
    # box2 = np.int0(box2)
    # cv2.drawContours(img , [box2] , 0 , (255 , 0 , 0) , 3)
    #
    #
    # cv2.imshow("img" , img)
    # cv2.waitKey(0)


    # img = cv2.imread("/Users/han/create_data/8-37-0-0-37-0-90.png")
    # if type(img) == type(None):
    #     sys.exit(1)
    #
    # json_info = parese_json("8-37-0-0-37-0-90.json")
    # pts_1 = json_info['shapes'][0]['points']
    # pts_2 = json_info['shapes'][1]['points']
    #
    #
    # img3 = img.copy()
    #
    #
    # array_pt1 = np.array(pts_1 , dtype=np.int32)
    # array_pt2 = np.array(pts_2 , dtype=np.int32)
    # array_pt1 = array_pt1.reshape((-1 , 1 , 2))
    # array_pt2 = array_pt2.reshape((-1 , 1 , 2))
    #
    # x0 , y0 , w0 , h0 = cv2.boundingRect(array_pt1)
    # x2 , y2 , w2 , h2 = cv2.boundingRect(array_pt2)
    #
    # # cv2.rectangle(img , (x0 , y0) , (x0 + w0 , y0 + h0) , (0 , 255 , 0) , 2)
    # cv2.rectangle(img , (x2 , y2) , (x2 + w2 , y2 + h2) , (255 , 0 , 0) , 2)
    #
    #
    # jiaodian = getPolygonCrossPoint(pts_1 , pts_2)
    #
    # print('pts_1:' , pts_1)
    # pts_1 = optimizerPolygon(jiaodian , pts_1)
    # print('pts_1:' , pts_1)
    #
    #
    # pts_1 = np.array(pts_1 , dtype=np.int32)
    # pts_1 = pts_1.reshape((-1 , 1, 2))
    #
    # x1 , y1 , w1 , h1 = cv2.boundingRect(pts_1)
    # cv2.rectangle(img , (x1 , y1) , (x1 + w1 , y1 + h1) , (0 , 0 , 255) , 2)
    # cv2.imshow("img" , img)
    # cv2.waitKey(0)



    img = np.zeros(shape=(600 , 600 , 3) , dtype=np.uint8)

    pts = [[50 , 50] , [150 , 150] , [150 , 50] , [50 , 150]]

    pts = ClockwiseSortpoints(pts)
    for i in pts:
        cv2.circle(img , (i[0] , i[1]) , 5 , (255 , 0 , 0) , 1)

    array_pts = np.array(pts , dtype=np.int32)
    array_pts = array_pts.reshape((-1 , 1 ,2))
    cv2.polylines(img , [array_pts] , True , (0 , 0 , 255) , 2)

    cv2.findContours()

    cv2.imshow("img" , img)
    cv2.waitKey(0)










































