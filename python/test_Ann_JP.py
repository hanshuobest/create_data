#coding:utf-8
import glob
import os

if __name__=='__main__':
	xml_lsts = glob.glob(os.path.dirname(os.getcwd()) + "/save/*.xml")
	jpg_lsts = glob.glob(os.path.dirname(os.getcwd()) + "/save/*.jpg")

	for jpg in jpg_lsts:
		xml_name = jpg.replace("jpg" , "xml")
		if os.path.exists(xml_name):
			continue
		else:
			os.remove(jpg)
