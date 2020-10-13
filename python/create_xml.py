#coding:utf-8
import cv2
import numpy as np
import glob
import os
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import XML_EXT
import multiprocessing
import time

def process(bmp_file):
    img = cv2.imread(bmp_file)
    jpg_file = bmp_file.replace("bmp" , "jpg")

    try:
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray , 0 , 255 , cv2.THRESH_BINARY)[1]
    except:
        os.remove(bmp_file)
        print('delete {}'.format(bmp_file))

    binary, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) >= 1:
        contours = sorted(contours, key=lambda item: len(item), reverse=True)
        for contour in contours:
            x , y , w , h = cv2.boundingRect(contour)
            if w < 50 or h < 50:
                continue

            rect = cv2.boundingRect(contour)

            writer = PascalVocWriter(os.getcwd(), os.path.basename(jpg_file), img.shape, localImgPath=jpg_file,
                                     usrname="auto")
            writer.verified = True
            writer.addBndBox(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], "166", 0)
            writer.save(targetFile=jpg_file[:-4] + XML_EXT)
    else:
        os.remove(jpg_file)


if __name__ == '__main__':
    bmp_lsts = glob.glob(os.getcwd() + "/*.bmp")

    start_time = time.time()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for i in bmp_lsts:
        pool.apply_async(process , (i,))
    pool.close()
    pool.join()

	
    print("finished")






