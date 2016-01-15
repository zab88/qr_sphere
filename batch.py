import numpy as np
import cv2, os, subprocess
from FF import FF

myDir = 'balls'
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

        if (file[-4:] != '.png'):
            continue

        black_cropped = FF.FF.black_and_white(cv2.imread(file_path))
        new_img = FF.FF.toSphere(black_cropped.copy())
        cv2.imwrite(tmpDir + os.path.sep + file, new_img)

        # rotate and apply distortion
        h_res, w_res = black_cropped.shape
        M = cv2.getRotationMatrix2D((w_res/2, h_res/2), 45, 1)
        black_rotated = cv2.warpAffine(black_cropped.copy(), M, (w_res, h_res))
        new_img_rotated = FF.FF.toSphere(black_rotated.copy())
        cv2.imwrite(tmpDir + os.path.sep + 'rot_'+file, new_img)

        qr_init = FF.FF.getQR(abs_file_path)
        qr_found = FF.FF.getQR(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + file)
        qr_rotated = FF.FF.getQR(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + 'rot_' + file)
        if qr_init:
            print(qr_init, 'init_' +file)
            files_found += 1
        elif qr_found:
            print(qr_found, file)
            files_found += 1
        elif qr_rotated:
            print(qr_rotated, 'rot_' + file)
            files_found += 1

print( str(files_found) + ' of ' + str(files_num) )