import numpy as np
import cv2, os, subprocess
from FF import FF

myDir = 'balls_old'
tmpDir = 'tmp'
files_num = 0
files_found = 0


for subdir, dirs, files in os.walk(myDir):
    for file in files:
        file_path = subdir + os.path.sep + file
        # abs_file_path = '"' + os.path.dirname(__file__)+'/'+file_path + '"'
        abs_file_path = os.path.dirname(__file__)+'/'+file_path
        # print( file_path, os.path.dirname(__file__) )
        files_num += 1

        if (file[-4:] != '.jpg'):
            continue

        new_img = FF.FF.black_and_white(cv2.imread(file_path))
        new_img = FF.FF.toSphere(new_img)
        cv2.imwrite(tmpDir + os.path.sep + file, new_img)

        qr_found = FF.FF.getQR(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + file)
        if qr_found:
            print(qr_found, file)
            files_found += 1

print( str(files_found) + ' of ' + str(files_num) )