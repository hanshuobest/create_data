#coding:utf-8
# statistic xml file,pick out the xml that have problem
# hanshuo
from __future__ import print_function
import os
import glob
import argparse
import multiprocessing
import time
import math


def union_dict(*objs):
    keys = set(sum([obj.keys() for obj in objs] , []))
    total = {}
    for key in keys:
        total[key] = sum([obj.get(key , 0) for obj in objs])
    return total

def readAnnotations(xml_path):
	import xml.etree.cElementTree as ET
	
	et = ET.parse(xml_path)
	element = et.getroot()
	element_objs = element.findall('object')
	# element_filename = element.find('filename').text
	element_width = int(element.find('size').find('width').text)
	element_height = int(element.find('size').find('height').text)
	
	
	results = []
	if element_width ==0 or element_height == 0:
		os.remove(xml_path)
		print('the xml size is 0 ,  delete the xml:' , xml_path)
		return results
	for element_obj in element_objs:
		result = []
		class_name = element_obj.find('name').text
		# print('class_name:' , class_name)
		if class_name == None:
			print('the xml class is None , delete the xml :' , os.path.basename(xml_path))
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

def statistic(xml_dir):
        xml_lsts = glob.glob(xml_dir + '/*.xml')
        num =  len(xml_lsts)
        results = []
        for i in xml_lsts:
                result = readAnnotations(i)
                if len(result) == 0:
                        os.remove(i)
                        print('the xml not annation ,  delete the xml:' , i)
                results.extend(result)
        # print(results)

        cls_dict = {}
        for i in results:
                if i[0] not in cls_dict.keys():
                        cls_dict[i[0]] = 1
                else:
                        cls_dict[i[0]] += 1

        print('-------------------------------------------------------------------')
        print('before sort the number of detect good:')
        cls_dict = sorted(cls_dict.items() , key = lambda item:int(item[0]))
        print(cls_dict)
        print('-------------------------------------------------------------------')

        cls_dict = dict(cls_dict)
        cls_dict = sorted(cls_dict.items() , key=lambda item:item[1] , reverse=True)
        print(cls_dict)
        print('the number of detect goods:' , len(cls_dict))
        print('-------------------------------------------------------------------')


def process_sub(xml_lsts):
	num = len(xml_lsts)
	results = []
	for i in xml_lsts:
		result = readAnnotations(i)
		if len(result) == 0:
			os.remove(i)
			print('the xml not annation ,  delete the xml:', i)
		results.extend(result)
	
	cls_dict = {}
	for i in results:
		if i[0] not in cls_dict.keys():
			cls_dict[i[0]] = 1
		else:
			cls_dict[i[0]] += 1
	
	#print('-------------------------------------------------------------------')
	#print('before sort the number of detect good:')
	#cls_dict = sorted(cls_dict.items(), key=lambda item: int(item[0]))
	#print('cls_dict:' ,cls_dict)
	#print('-------------------------------------------------------------------')
	
	return dict(cls_dict)


if __name__ == '__main__':
	xml_lsts = glob.glob(os.path.dirname(os.getcwd()) + "/save/*.xml")
	num_process = 2
	
	start_time = time.time()	
	per_count = int(math.ceil(float(len(xml_lsts))/num_process))
	print(per_count)
	process_lst = []
	pool = multiprocessing.Pool(processes=num_process)
	
	results_total = []
	for i in range(num_process):
		result = None
		if i < num_process - 1:
			result = pool.apply_async(process_sub , (xml_lsts[i * per_count: (i + 1) * per_count],))
		else:
			result = pool.apply_async(process_sub , (xml_lsts[i * per_count:],))
			
		results_total.append(result.get())
	pool.close()
	pool.join()
	
	# print('results_total:' , results_total)
	total_dict = union_dict(results_total[0] , results_total[1])
	print('-------------------------------------------------------------------')
        print('before sort the number of detect good:')
        total_dict = sorted(total_dict.items() , key = lambda item:int(item[0]))
        print(total_dict)
        print('-------------------------------------------------------------------')

        total_dict = dict(total_dict)
        total_dict = sorted(total_dict.items() , key=lambda item:item[1] , reverse=True)
        print(total_dict)
        print('the number of detect goods:' , len(total_dict))
        print('-------------------------------------------------------------------')
	
	cost_time = time.time() - start_time
	print('cost time:' , cost_time)

