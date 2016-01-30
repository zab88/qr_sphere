import cv2
import numpy as np
import subprocess, os
import ConfigParser
from scipy.interpolate import griddata
import FF

class SearcherSimplest:
    abs_img_path = None
    file_name = None
    result = None

    tmpDir = 'tmp_simplest'

    def __init__(self, abs_img_path, file_name):
        self.abs_img_path = abs_img_path
        self.file_name = file_name
        self.tmpDir = os.path.dirname(__file__)+os.path.sep +'..'+os.path.sep+ self.tmpDir

    def search(self):
        black_cropped = FF.FF.black_and_white(cv2.imread(self.abs_img_path))
        new_img = FF.FF.toSphere(black_cropped.copy())

        # print(self.tmpDir + os.path.sep + self.file_name)
        cv2.imwrite(self.tmpDir + os.path.sep + self.file_name, new_img)

        qr_init = FF.FF.getQR(self.abs_img_path)
        qr_found = FF.FF.getQR(self.tmpDir+os.path.sep + self.file_name)
        # qr_found = FF.FF.getQR2(self.tmpDir+os.path.sep + file)
        if qr_init and qr_init != 'NOT FOUND\r\n':
            return qr_init
        elif qr_found and qr_found != 'NOT FOUND\r\n':
            return qr_found

        return None

