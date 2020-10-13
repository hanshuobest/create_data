#coding:utf-8
import os
import glob


if __name__ == '__main__':
    shape_dir = os.path.join(os.path.dirname(os.getcwd()) , "shapes")
    print(shape_dir)

    f_mask = open(os.path.join(shape_dir , "mask.txt") , "w")
    f_foreground = open(os.path.join(shape_dir , "foreground.txt") , "w")

    png_lsts = glob.glob(shape_dir + "/*.png")
    for png in png_lsts:
        if "mask" in png:
            f_mask.write(png)
            f_mask.write("\n")
        else:
            f_foreground.write(png)
            f_foreground.write("\n")

    f_mask.close()
    f_foreground.close()