import cv2
import numpy as np
import subprocess, os

class FF:
    # zbar = 'C:/Program Files (x86)/ZBar/bin/zbarimg.exe'
    zbar = os.path.dirname(__file__)+os.path.sep + '/../bin'+os.path.sep+'zbarimg.exe'

    def __init__(self):
        pass

    @staticmethod
    def black_and_white(img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        img_cropped = FF.crop_img2(img_grey)
        img_bin = cv2.threshold(img_cropped, 127, 255, cv2.THRESH_OTSU)[1]
        # img_bin = cv2.threshold(img_cropped, 100, 255, cv2.THRESH_BINARY)[1]
        return img_bin

    @staticmethod
    def crop_img(img_origin):
        h_origin, w_origin = img_origin.shape
        return img_origin[60:h_origin-71, 89:w_origin-113]

    @staticmethod
    def crop_img2(img_origin):
        h_origin, w_origin = img_origin.shape
        return img_origin[150:h_origin-220, 670:w_origin-580]

    @staticmethod
    def toSphere(src):
        h, w = src.shape[0:2]
        # print(h, w, 'h, w of cropped')

        intrinsics = np.zeros((3, 3), np.float64)


        intrinsics[0, 0] = 8500
        # intrinsics[0, 1] = -2000.0
        # intrinsics[1, 0] = 2000.0
        intrinsics[1, 1] = 8500
        intrinsics[2, 2] = 1.0
        intrinsics[0, 2] = w/2.
        intrinsics[1, 2] = h/2.
        # print(intrinsics)

        newCamMtx = np.zeros((3, 3), np.float64)
        newCamMtx[0, 0] = 7000
        # intrinsics[0, 1] = -2000.0
        # intrinsics[1, 0] = 2000.0
        newCamMtx[1, 1] = 7000
        newCamMtx[2, 2] = 1.0
        newCamMtx[0, 2] = w/2.
        newCamMtx[1, 2] = h/2.

        dist_coeffs = np.zeros((1, 4), np.float64)
        dist_coeffs[0, 0] = -70.0
        dist_coeffs[0, 1] = 0.0
        dist_coeffs[0, 2] = 0.0
        dist_coeffs[0, 3] = 0.0
        # dist_coeffs = np.array([[float(k1)],[float(k2)],[float(p1)], [float(p2)]], np.float64)
        # print dist_coeffs

        map1, map2 = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, newCamMtx, src.shape[:2], cv2.CV_16SC2)
        #cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]])
        res = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)

        return res

    @staticmethod
    def getQR(abs_file_path):
        result = None
        try:
            # print('"'+FF.zbar+'" -q "' + abs_file_path+'"')
            out = subprocess.check_output('"'+FF.zbar+'" -q "' + abs_file_path+'"', shell=True, stderr=subprocess.STDOUT)
            # files_found += 1
            # print(file_path, out)
            result = out
        except subprocess.CalledProcessError as e:
            out = e.output

        return result