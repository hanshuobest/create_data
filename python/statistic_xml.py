# coding:utf-8
# statistic xml file,pick out the xml that have problem
# hanshuo

from __future__ import print_function
import os
import glob
import argparse
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


def readAnnotations(xml_path):
    import xml.etree.cElementTree as ET

    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')
    # element_filename = element.find('filename').text
    element_width = int(element.find('size').find('width').text)
    element_height = int(element.find('size').find('height').text)

    results = []
    if element_width == 0 or element_height == 0:
        os.remove(xml_path)
        print('the xml size is 0 ,  delete the xml:', xml_path)
        return results
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text
        # print('class_name:' , class_name)
        if class_name == None:
            print('the xml class is None , delete the xml :',
                  os.path.basename(xml_path))
        else:
            obj_bbox = element_obj.find('bndbox')
            x1 = int(round(float(obj_bbox.find('xmin').text)))
            y1 = int(round(float(obj_bbox.find('ymin').text)))
            x2 = int(round(float(obj_bbox.find('xmax').text)))
            y2 = int(round(float(obj_bbox.find('ymax').text)))

            result.append(class_name)
            result.append(x1)
            result.append(y1)
            result.append(x2)
            result.append(y2)

            results.append(result)

    return results


def statistic(logo_dir):
    # xml_lsts = glob.glob(xml_dir + '/*.xml')
    # num = len(xml_lsts)
    # results = []
    # for i in xml_lsts:
    #     result = readAnnotations(i)
    #     if len(result) == 0:
    #         os.remove(i)
    #         print('the xml not annation ,  delete the xml:', i)
    #     results.extend(result)
    # print(results)
    
    dir_lst = os.listdir(logo_dir)
    cls_dict = {}
    for d in dir_lst:
        label_cls = d.split("_")[0]
        if label_cls not in cls_dict:
            cls_dict[label_cls] = 1
        else:
            cls_dict[label_cls] += 1
    
    # for i in results:
    #     if i[0] not in cls_dict.keys():
    #         cls_dict[i[0]] = 1
    #     else:
    #         cls_dict[i[0]] += 1


    cls_dict = dict(cls_dict)
    cls_dict = sorted(cls_dict.items(), key=lambda item: item[1], reverse=True)
    print(cls_dict)
    print('the number of detect objects:', len(cls_dict))
    print('-------------------------------------------------------------------')

    labels = []
    sizes = []
    explode = []
    
    for d in cls_dict:
        labels.append(d[0])
        sizes.append(d[1])
        explode.append(0.0)
        
        
    plt.pie(sizes , labels=labels , explode=explode , autopct='%1.1f%%' , shadow=False , startangle=150)
    plt.title("classes distribution")
    plt.axis('equal')
    plt.legend(loc="upper right",fontsize=10,bbox_to_anchor=(1.1,1.05),borderaxespad=0.3)
    
    plt.show()


if __name__ == '__main__':
    sku_logo_dir = "../sku/sku_logo/"
    # xml_dir = os.getcwd()
    start_time = time.time()
    statistic(sku_logo_dir)
    cost_time = time.time() - start_time
    print('cost time:', cost_time)

