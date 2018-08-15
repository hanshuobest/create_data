#coding:utf-8
import cv2
import json
import os
import numpy as np
import glob
import shutil
import sys

def parese_json(fpath):
    with open(fpath, 'rb') as f:
        myjson = json.load(f)
        # imagePath = myjson["imagePath"]
        # print(imagePath)
        # print('shapes:' , myjson["shapes"])
        # print('points:' , myjson["shapes"][0]['points'])
        # print('label:'  , myjson["shapes"][0]['label'])
        return myjson


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
    json_lst = glob.glob(os.path.join(os.getcwd() , "sku") + "/*.json")
    img_lst =  glob.glob(os.path.join(os.getcwd() , "sku") + "/*.png")

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

    for j_info in json_lst:
        json_info = parese_json(j_info)
        polygon_points = json_info["shapes"][0]["points"]
        label = json_info["shapes"][0]["label"]
        imagePath = os.path.join(os.getcwd() , "sku") + "/" + os.path.basename(json_info['imagePath'])

        img = cv2.imread(imagePath)
        if type(img) == type(None):
            print "打开图片失败！"
            break
        img_h , img_w , _ = img.shape
        pts = np.array(polygon_points , dtype=np.int32)
        x , y , w , h = cv2.boundingRect(pts)
        # cv2.rectangle(img , (x , y) , (x + w , y + h) , (255 , 0 , 0) , thickness=2)
        for i in range(y , y + h):
            for j in range(x , x + w):

                is_flag = isInPolygon_2(polygon_points , [j , i])
                # print('flag:' , is_flag)
                if is_flag:
                    continue
                else:
                    img[i , j] = [0 , 0 , 0]

        imgROI = img[y : y + h , x : x + w , :]
        roi_polygon_points = []
        for pp in polygon_points:
            roi_polygon_points.append([pp[0] - x , pp[1] - y])

        roi_json = {}
        roi_json["points"] = roi_polygon_points

        roi_imagePath = roi_path + "/" + os.path.basename(json_info['imagePath'])[:-4] + ".png"
        cv2.imwrite(roi_imagePath , imgROI)
        roi_jsonPath = roi_path + "/" + os.path.basename(json_info['imagePath'])[:-4] + "-" + "roi.json"
        with open(roi_jsonPath , 'w') as json_file:
            json.dump(roi_json , json_file , ensure_ascii=False)

    cv2.waitKey(0)


    # roi = cv2.imread("/Users/han/Downloads/bn_exp/sku/roi/42-0.png")
    # if type(roi) == type(None):
    #     print '打开图片失败'
    #
    # json_info = parese_json("/Users/han/Downloads/bn_exp/sku/roi/42-0-roi.json")
    #
    # pts = np.array(json_info['points'] , dtype=np.int32)
    # pts = pts.reshape((-1 , 1 , 2))
    #
    # rotate_img , rotate_matrix = rotate_image(roi , 0)
    # print('matrix:' , np.shape(rotate_matrix))
    #
    # # 保存旋转后的多边形坐标
    # rotate_pts = []
    # for i in range(pts.shape[0]):
    #     tmp = np.append(pts[i] , [1]).reshape(3 , 1)
    #     print('tmp:' , tmp.shape)
    #
    #     x_y = np.matmul(rotate_matrix , tmp)
    #     print('x_y:' , x_y.shape)
    #     rotate_pts.append([x_y[0][0] , x_y[1][0]])
    # print(rotate_pts)
    #
    # # roi = cv2.polylines(roi, [pts], True, (0, 0, 255))
    #
    # rotate_pts = np.array(rotate_pts , dtype=np.int32)
    # rotate_pts = rotate_pts.reshape((-1 , 1 , 2))
    # rotate_img = cv2.polylines(rotate_img , [rotate_pts] , True , (0 , 0 , 255))
    #
    #
    #
    #
    #
    # cv2.imshow("roi" , roi)
    # cv2.imshow("rotate_img" , rotate_img)
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















