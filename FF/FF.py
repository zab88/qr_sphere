import cv2
import numpy as np
import subprocess, os
import ConfigParser
from scipy.interpolate import griddata

class FF:
    # zbar = 'C:/Program Files (x86)/ZBar/bin/zbarimg.exe'
    zbar = os.path.dirname(__file__)+os.path.sep + '/../bin'+os.path.sep+'zbarimg.exe'
    zxing = os.path.dirname(__file__)+os.path.sep + '/../zxing'+os.path.sep+'Bingo.Cli.Recognizer.exe'

    def __init__(self):
        pass

    @staticmethod
    def readSettings():
        Config = ConfigParser.ConfigParser()
        Config.read("settings.ini")
        FF.crop_top = Config.getint('Crop_balls2', 'crop_top')
        FF.crop_left = Config.getint('Crop_balls2', 'crop_left')
        FF.crop_bottom = Config.getint('Crop_balls2', 'crop_bottom')
        FF.crop_right = Config.getint('Crop_balls2', 'crop_right')

    @staticmethod
    def black_and_white(img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        img_blur = cv2.blur(img_grey, (5, 5))
        img_cropped = FF.crop_img4(img_blur)
        img_bin = cv2.threshold(img_cropped, 127, 255, cv2.THRESH_OTSU)[1]
        # img_bin = cv2.threshold(img_cropped, 100, 255, cv2.THRESH_BINARY)[1]
        # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, (3,3), iterations=1)
        # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, (13,13), iterations=5)
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
    def crop_img3(img_origin):
        h_origin, w_origin = img_origin.shape[:2]
        return img_origin[FF.crop_top:h_origin-FF.crop_bottom, FF.crop_left:w_origin-FF.crop_right]

    @staticmethod
    def crop_img4(img_origin):
        h_origin, w_origin = img_origin.shape[:2]
        return img_origin[150:h_origin-170, 600:w_origin-560]

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

    @staticmethod
    def getQR2(abs_file_path):
        result = None
        try:
            # print('"'+FF.zbar+'" -q "' + abs_file_path+'"')
            out = subprocess.check_output('"'+FF.zxing+'" zxing "' + abs_file_path+'"', shell=True, stderr=subprocess.STDOUT)
            # files_found += 1
            # print(file_path, out)
            result = out
        except subprocess.CalledProcessError as e:
            out = e.output

        return result

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return int(x), int(y)

    @staticmethod
    def getWarpMx(s1, s2, s3, cent, corner_num):
        far1, far2, far3, cl1, cl2, cl3 = 0, 0, 0, 99999, 99999, 99999
        farP1, farP2, farP3 = None, None, None
        clP1, clP2, clP3 = None, None, None
        unP1, unP2, unP4 = None, None, None #closest points to unknown
        pp_edge = None
        for p in s1:
            cur_dist = np.linalg.norm(p-cent)
            if cur_dist > far1:
                far1 = cur_dist
                farP1 = p
            if cur_dist < cl1:
                cl1 = cur_dist
                clP1 = p

        for p in s2:
            cur_dist = np.linalg.norm(p-cent)
            if cur_dist > far2:
                far2 = cur_dist
                farP2 = p
            if cur_dist < cl2:
                cl2 = cur_dist
                clP2 = p

        for p in s3:
            cur_dist = np.linalg.norm(p-cent)
            if cur_dist > far3:
                far3 = cur_dist
                farP3 = p
            if cur_dist < cl3:
                cl3 = cur_dist
                clP3 = p

        if corner_num == 1:
            pp_edge = farP1
            SS1 = s2; SS2 = s3; SS0 = s1;
            pp_edge1 = farP2; pp_edge2 = farP3
            pp_in1=clP2; pp_in2=clP3; pp_in0=clP1
        if corner_num == 2:
            pp_edge = farP2
            SS1 = s1; SS2 = s3; SS0 = s2;
            pp_edge1 = farP1; pp_edge2 = farP3
            pp_in1=clP1; pp_in2=clP3; pp_in0=clP2
        if corner_num == 3:
            pp_edge = farP3
            SS1 = s1; SS2 = s2; SS0 = s3;
            pp_edge1 = farP1; pp_edge2 = farP2
            pp_in1=clP1; pp_in2=clP2; pp_in0=clP3

        #Searching unP1 and unP2
        dist_max = 0
        for p in SS1:
            cur_dist = np.linalg.norm(p-pp_edge)
            if (p[0][0]==farP1[0][0] and p[0][1]==farP1[0][1]) or (p[0][0]==farP2[0][0] and p[0][1]==farP2[0][1]) or (p[0][0]==farP3[0][0] and p[0][1]==farP3[0][1]):
                continue
            if cur_dist > dist_max:
                dist_max = cur_dist
                unP1 = p
        dist_max = 0
        for p in SS2:
            cur_dist = np.linalg.norm(p-pp_edge)
            if (p[0][0]==farP1[0][0] and p[0][1]==farP1[0][1]) or (p[0][0]==farP2[0][0] and p[0][1]==farP2[0][1]) or (p[0][0]==farP3[0][0] and p[0][1]==farP3[0][1]):
                continue
            if cur_dist > dist_max:
                dist_max = cur_dist
                unP2 = p

        #searching last on SS1 and on SS2
        for p in SS1:
            if (p[0][0]==pp_edge1[0][0] and p[0][1]==pp_edge1[0][1]) or (p[0][0]==unP1[0][0] and p[0][1]==unP1[0][1]) or (p[0][0]==pp_in1[0][0] and p[0][1]==pp_in1[0][1]):
                continue
            pp_bok10 = p
        for p in SS2:
            if (p[0][0]==pp_edge2[0][0] and p[0][1]==pp_edge2[0][1]) or (p[0][0]==unP2[0][0] and p[0][1]==unP2[0][1]) or (p[0][0]==pp_in2[0][0] and p[0][1]==pp_in2[0][1]):
                continue
            pp_bok20 = p

        # searching sides of zero
        dist_min1 = 99999
        dist_min2 = 99999
        for p in SS0:
            cur_dist1 = np.linalg.norm(p-pp_bok10)
            cur_dist2 = np.linalg.norm(p-pp_bok20)
            if (p[0][0]==pp_in0[0][0] and p[0][1]==pp_in0[0][1]) or (p[0][0]==pp_edge[0][0] and p[0][1]==pp_edge[0][1]):
                continue
            if cur_dist1 < dist_min1:
                dist_min1 = cur_dist1
                pp_bok01 = p
            if cur_dist2 < dist_min2:
                dist_min2 = cur_dist2
                pp_bok02 = p

        # getting unknown vertex
        ux, uy = FF.line_intersection((list(unP1[0]), list(pp_edge1[0])), (list(unP2[0]), list(pp_edge2[0])))
        unP4 = [ux, uy]

        # source = np.array([[farP1[0][0], farP1[0][1]], [farP2[0][0], farP2[0][1]], [farP3[0][0], farP3[0][1]]])
        source_draw = np.array([[pp_edge1[0][0], pp_edge1[0][1]],
                           [pp_edge[0][0], pp_edge[0][1]],
                           [pp_edge2[0][0], pp_edge2[0][1]],
                           [ux, uy],
                           [unP1[0][0], unP1[0][1]],
                           [unP2[0][0], unP2[0][1]]
                           ])
        # print(source)
        source = np.array([[pp_edge1[0][1], pp_edge1[0][0]],
                           [pp_edge[0][1], pp_edge[0][0]],
                           [pp_edge2[0][1], pp_edge2[0][0]],
                           [uy, ux],
                           [unP1[0][1], unP1[0][0]],
                           [unP2[0][1], unP2[0][0]],
                           [pp_in1[0][1], pp_in1[0][0]],
                           [pp_in0[0][1], pp_in0[0][0]],
                           [pp_in2[0][1], pp_in2[0][0]],
                           [pp_bok10[0][1], pp_bok10[0][0]],
                           [pp_bok20[0][1], pp_bok20[0][0]],
                           [pp_bok01[0][1], pp_bok01[0][0]],
                           [pp_bok02[0][1], pp_bok02[0][0]]
                           ])
        sources = []
        for xx in range(-20, 20, 10):
            for yy in range(-20, 20, 10):
                ux_new = ux+xx
                uy_new = uy+yy
                source_new = source
                source_new[3] = [uy_new, ux_new]
                sources.append(source_new.copy())

        # 4 pixels for squire, so 84=4*21
        # destination = np.array([[0,0], [0,99], [0,199],
        #           [99,0],[99,99],[99,199],
        #           [199,0],[199,99],[199,199]])
        destination = np.array([[0,0], [0,83], [83,83], [83,0], [28,0], [83,56], [28,28], [28,56], [56,56],
                                [0, 28],
                                [56, 83],
                                [0, 56],
                                [28, 83]
                                ])
        # source = np.array([[43, 55], [166,46],[274,285]])

        return sources, destination, source_draw

    @staticmethod
    def getRightQr( source, destination, img):
        grid_x, grid_y = np.mgrid[0:83:84j, 0:83:84j]
        grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
        map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(84,84)
        map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(84,84)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)#cv2.INTER_LINEAR

        return warped