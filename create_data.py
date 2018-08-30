#coding:utf-8
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




if __name__ == '__main__':
    # 生成模板图像
    print '开始生成ROI模板'
    json_lst = glob.glob(os.path.join(os.getcwd(), "sku") + "/*.json")
    img_lst = glob.glob(os.path.join(os.getcwd(), "sku") + "/*.png")
    print('num of json:' , len(json_lst))
    print('num of img_lst:' , len(img_lst))

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
    # print(len(json_lst))
    for j_info in json_lst:
	# print(os.path.basename(j_info))
        json_info = parese_json(j_info)
        polygon_points = json_info["shapes"][0]["points"]
        label = json_info["shapes"][0]["label"]
        imagePath = os.path.join(os.getcwd(), "sku") + "/" + os.path.basename(json_info['imagePath'])
	# print('imagePath:' , imagePath)
	# print('imagePath:' , imagePath)
        img = cv2.imread(imagePath)
        if type(img) == type(None):
            print "打开图片失败！"
            continue
        img_h, img_w, _ = img.shape
        pts = np.array(polygon_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        # cv2.rectangle(img , (x , y) , (x + w , y + h) , (255 , 0 , 0) , thickness=2)
        for i in range(y, y + h):
            for j in range(x, x + w):

                is_flag = isInPolygon_2(polygon_points, [j, i])
                # print('flag:' , is_flag)
                if is_flag:
                    continue
                else:
                    img[i, j] = [0, 0, 0]

        imgROI = img[y: y + h, x: x + w, :]
        roi_polygon_points = []
        for pp in polygon_points:
            roi_polygon_points.append([pp[0] - x, pp[1] - y])

        roi_json = {}
        roi_json["points"] = roi_polygon_points

        roi_imagePath = roi_path + "/" + os.path.basename(json_info['imagePath'])[:-4] + ".png"
        cv2.imwrite(roi_imagePath, imgROI)
        roi_jsonPath = roi_path + "/" + os.path.basename(json_info['imagePath'])[:-4] + "-" + "roi.json"
	# print(roi_jsonPath)
        with open(roi_jsonPath, 'w') as json_file:
            json.dump(roi_json, json_file, ensure_ascii=False)
    print '模板ROI生成完毕！'

    background_imgs = glob.glob(os.path.join(os.getcwd() , "sku" , "background") + "/*.png")
    background_num = len(background_imgs)
    if background_num == 0:
        print '没有背景图片'
        sys.exit(1)

    roi_imgs = glob.glob(os.path.join(os.getcwd() , "sku" , "roi") + "/*.png")
    roi_num = len(roi_imgs)
    if roi_num <= 1:
        print '没有足够的roi图片'
        sys.exit(1)
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

    rotate_angles = [0 , 30 , 60 , 90 , 120 , 150]
    for i in tqdm(range(len(background_imgs))):
        img = cv2.imread(background_imgs[i])
        img_h , img_w , _ = img.shape


        pt1_x = np.random.randint(0 , img_w)
        pt1_y = np.random.randint(0 , img_h)


        roi_index1 = np.random.choice(range(roi_num) , int(roi_num * 0.2) , replace=False)

        for j in roi_index1: # 第一次遍历roi
            result = img.copy()
            roi_img_1 = cv2.imread(roi_imgs[j])
            if type(roi_img_1) == type(None):
                print '读取roi图片失败'
                break
                # 获取roi的多边形坐标
            poly_roi_img1 = parese_json(roi_imgs[j][:-4] + "-roi.json")['points']
            label_1 = os.path.basename(roi_imgs[j]).split('-')[0]

            # 图片缩放
            if np.random.randint(0 , 2):
                roi_img_1 , poly_roi_img1 = imageZoom(roi_img_1 , poly_roi_img1 , random.randint(0 , 1))

            rotate_angles_index1 = np.random.choice(rotate_angles , 1 , replace=False)
            # 遍历旋转角度
            for j_angle in rotate_angles_index1:
                result2 = result.copy()
                rotate_roi_img_1 , rotate_matrix_1 = rotate_image(roi_img_1 , j_angle)
                rotate_roi_h_1, rotate_roi_w_1, _ = rotate_roi_img_1.shape

                if (pt1_x + rotate_roi_w_1) > img_w:
                    pt1_x = img_w - rotate_roi_w_1
                if (pt1_y + rotate_roi_h_1) > img_h:
                    pt1_y = img_h - rotate_roi_h_1
                poly_roi_img1 = np.array(poly_roi_img1 , dtype=np.int32)
                # 根据仿射变换计算旋转后的轮廓点
                rotate_poly_roi_img1 = warfPoints(poly_roi_img1 , rotate_matrix_1)
                update_PolygonPoints(rotate_poly_roi_img1, [pt1_x, pt1_y])
                rotate_roi_1 = result2[pt1_y: pt1_y + rotate_roi_h_1, pt1_x: pt1_x + rotate_roi_w_1, :]

                for i_1 in range(rotate_roi_h_1):
                    for j_1 in range(rotate_roi_w_1):
                        if int(np.sum(rotate_roi_img_1[i_1][j_1])) <= 20:
                            continue
                        else:
                            rotate_roi_1[i_1][j_1] = rotate_roi_img_1[i_1][j_1]

                result2[pt1_y: pt1_y + rotate_roi_h_1, pt1_x: pt1_x + rotate_roi_w_1, :] = rotate_roi_1

                # json_img_angle_points = {}
                # json_img_angle_points["points"] = rotate_poly_roi_img1
                # json_img_angle = generate_dir + "/" + os.path.basename(i)[:-4] + "-" +  os.path.basename(roi_imgs[j])[:-4] + "-" + str(j_angle) + ".json"
                # print('json_img_angle:' , json_img_angle)
                # with open(json_img_angle , 'w') as json_file:
                #     json.dump(json_img_angle_points , json_file , ensure_ascii=False)


                roi_index2 = np.random.choice(range(roi_num) , int(0.2 * roi_num) , replace=False)
                for k in roi_index2: # 第二次遍历roi
                    roi_img_2 = cv2.imread(roi_imgs[k])
                    if type(roi_img_2) == type(None):
                        print '读取roi图片失败'
                        break


                    poly_roi_img2 = parese_json(roi_imgs[k][:-4] + "-roi.json")['points']
                    label_2 = os.path.basename(roi_imgs[k]).split('-')[0]
                    result3 = result2.copy()

                    if np.random.randint(0 , 2):
                        roi_img_2 , poly_roi_img2 = imageZoom(roi_img_2 , poly_roi_img2 , random.randint(0 , 1))

                    rotate_angles_index2 = np.random.choice(rotate_angles, 1, replace=False)
                    for k_angle in rotate_angles_index2:
                        rotate_roi_img_2 , rotate_matrix_2 = rotate_image(roi_img_2 , k_angle)
                        rotate_roi_h_2 , rotate_roi_w_2 , _ = rotate_roi_img_2.shape

                        pt2_x = np.random.randint(pt1_x - rotate_roi_w_1 , pt1_x + rotate_roi_w_1)
                        pt2_y = np.random.randint(pt1_y - rotate_roi_h_2, pt1_y + rotate_roi_h_1)

                        if (pt2_x + rotate_roi_w_2) > img_w:
                            pt2_x = img_w - rotate_roi_w_2
                        if (pt2_y + rotate_roi_h_2) > img_h:
                            pt2_y = img_h - rotate_roi_h_2
                        if pt2_y < 0:
                            pt2_y = 0
                        if pt2_x <= 0:
                            pt2_x = 0

                        poly_roi_img2 = np.array(poly_roi_img2 , dtype=np.int32)
                        rotate_poly_roi_img2 = warfPoints(poly_roi_img2 , rotate_matrix_2)
                        update_PolygonPoints(rotate_poly_roi_img2, [pt2_x, pt2_y])

                        # json_img_angle_points_2 = {}
                        # json_img_angle_points_2['points'] = rotate_poly_roi_img2
                        # json_img_angle_points_2_path = generate_dir + "/" + os.path.basename(i)[:-4] + "-" + os.path.basename(roi_imgs[j])[:-4] + "-" + str(j_angle) + "-" +  os.path.basename(roi_imgs[k])[:-4] + "-" + str(k_angle) + ".json"
                        # with open(json_img_angle_points_2_path , "w") as json_file:
                        #     json.dump(json_img_angle_points_2 , json_file , ensure_ascii=False)


                        result4 = result3.copy()
                        rotate_roi_2 = result4[pt2_y: pt2_y + rotate_roi_h_2, pt2_x: pt2_x + rotate_roi_w_2, :]

                        for i_2 in range(rotate_roi_h_2):
                            for j_2 in range(rotate_roi_w_2):
                                if int(np.sum(rotate_roi_img_2[i_2][j_2])) <= 20:
                                    continue
                                else:
                                    rotate_roi_2[i_2][j_2] = rotate_roi_img_2[i_2][j_2]

                        result4[pt2_y:pt2_y + rotate_roi_h_2, pt2_x: pt2_x + rotate_roi_w_2, :] = rotate_roi_2

                        jiaodian = getPolygonCrossPoint(rotate_poly_roi_img1, rotate_poly_roi_img2)
                        if len(jiaodian) == 0:
                            continue

                        # print('jiaodian length:' , len(jiaodian))
                        ClockwiseSortpoints(jiaodian)

                        # jiaodian_array = np.array(jiaodian , dtype=np.int32)
                        # jiaodian_array =  jiaodian_array.reshape((-1 , 1 , 2))
                        # cv2.polylines(result4 , [jiaodian_array] , True , (0 , 0 , 255) , 2)


                        poly1, poly2 = updatePolygon(rotate_poly_roi_img1, rotate_poly_roi_img2, jiaodian)

                        poly1 = np.array(poly1, dtype=np.int32)
                        poly1 = poly1.reshape((-1, 1, 2))
                        poly2 = np.array(poly2, dtype=np.int32)
                        poly2 = poly2.reshape((-1, 1, 2))

                        x1, y1, w1, h1 = cv2.boundingRect(poly1)
                        x2, y2, w2, h2 = cv2.boundingRect(poly2)

                        # cv2.rectangle(img , (x1 , y1) , (x1 + w1 , y1 + h1) , (0 , 0 , 255) , 1)
                        # cv2.rectangle(img , (x2 , y2) , (x2 + w2 , y2 + h2) , (0 , 255 , 0) , 1)

                        iou = maxIou(x1 + 0.5 * w1, y1 + 0.5 * h1, w1, h1, x2 + 0.5 * w2, y2 + 0.5 * h2, w2, h2)
                        if iou < 0.1:
                            continue
                            # print '没有重叠，过滤掉'
                        elif iou > 0.5:
                            # print '两个bounding box重叠度为：', iou
                            # print '将舍弃该张图像'
                            continue




                        imgFileName = os.path.basename(background_imgs[i])[:-4] + "-" + os.path.basename(roi_imgs[j])[:-4] + "-" + str(j_angle) + "-" + os.path.basename(roi_imgs[k][:-4]) + "-" + str(k_angle) + ".png"
                        imagePath = os.path.join(imgFolderName, imgFileName)
                        writer = PascalVocWriter(imgFolderName, imgFileName, (img_h, img_w, 3), localImgPath=imagePath,
                                                 usrname="auto")
                        writer.verified = True
                        writer.addBndBox(x1, y1, x1 + w1, y1 + h1, label_1, 0)
                        writer.addBndBox(x2, y2, x2 + w2, y2 + h2, label_2, 0)
                        writer.save(targetFile=imagePath[:-4] + XML_EXT)

                        if np.random.randint(1 , 11) == 10:
                            result4 = addPepperNoise(result4)
                        if np.random.randint(1 , 6) == 3:
                            result4 = cv2.GaussianBlur(result4 , (5 , 5) , 0)


                        cv2.imwrite(imagePath, result4)

    cv2.waitKey(0)






