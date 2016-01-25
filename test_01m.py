import cv2, os, math
from scipy.interpolate import griddata
import numpy as np
import itertools
from FF import FF

def isQrVertex(p1, p2, p3):
    dist12 = np.linalg.norm(p1-p2)
    dist13 = np.linalg.norm(p1-p3)
    dist23 = np.linalg.norm(p2-p3)

    k1, k2, k3, k4 = 0.9, 1.1, 1.25, 1.55
    if (dist12*k1 < dist13 < dist12*k2) and (dist13*k3<dist23<dist13*k4):
        return True, 1, dist23
    if (dist23*k1 < dist12 < dist23*k2) and (dist12*k3<dist13<dist12*k4):
        return True, 2, dist13
    if (dist13*k1 < dist23 < dist13*k2) and (dist23*k3<dist12<dist23*k4):
        return True, 3, dist12

    return False, 0, 99999

myDir = 'balls3'
tmpDir = 'tmp'
found_num = 0
FF.FF.readSettings()

for subdir, dirs, files in os.walk(myDir):
    for file in files:
        file_path = subdir + os.path.sep + file
        abs_file_path = os.path.dirname(__file__)+'/'+file_path

        if (file[-4:] != '.png' and file[-4:] != '.jpg'):
            continue

        all_squires = []
        all_centers = []
        img = cv2.imread(abs_file_path)
        #crop
        img = FF.FF.crop_img4(img)
        #white balance
        img = FF.FF.adjust_gamma(img, 1.5)
        #blur
        img = cv2.blur(img, (3,3))
        # sphere warping
        img = FF.FF.toSphere(img.copy())
        # to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret,thresh = cv2.threshold(gray,127,255,1)
        # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)[1]
        #adding borders
        h, w = thresh.shape
        # cv2.line(img, (0, 0), (w, 0), (255, 127, 0), 21)
        # cv2.line(img, (w, 0), (w, h), (255, 127, 0), 21)
        # cv2.line(img, (w, h), (0, h), (255, 127, 0), 21)
        # cv2.line(img, (0, h), (0, 0), (255, 127, 0), 21)
        cv2.line(thresh, (0, 0), (w, 0), (255), 31)
        cv2.line(thresh, (w, 0), (w, h), (255), 31)
        cv2.line(thresh, (w, h), (0, h), (255), 31)
        cv2.line(thresh, (0, h), (0, 0), (255), 31)

        # aa = thresh.copy()
        # thresh = aa.copy()
        # cv2.imshow("ll", thresh)
        # cv2.waitKey(0)
        thresh = 255 - thresh

        contours,hhh_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours,h = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # contours,h = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            # cv2.drawContours(img,[cnt],0,(255, 255, 0), 1)
            # continue
            approx = cv2.approxPolyDP(cnt,0.017*cv2.arcLength(cnt,True),True)
            if len(approx)==4 :
                approx_area = cv2.contourArea(approx)
                x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(approx)
                # print(approx_area, w_cnt, h_cnt)
                if approx_area > 100 and np.float32(w_cnt)/np.float32(h_cnt)>0.6 and np.float32(w_cnt)/np.float32(h_cnt)<1.6:
                    # cv2.drawContours(img,[cnt],0,(0,0,255), 3)#-1
                    cv2.drawContours(img,[approx],0,(0, 255, 0), 3)#-1

                    #center search
                    M = cv2.moments(approx)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    # all_centers.append([cy, cx])
                    all_centers.append(np.array([cx, cy], dtype=np.int32))
                    all_squires.append(approx)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 255), 3)

        #we have
        for i in itertools.permutations(range(0, len(all_centers) ), 3):
            isQr, corner_num, diag_length = isQrVertex(all_centers[i[0]], all_centers[i[1]], all_centers[i[2]])
            # print(diag_length, h, w)
            if isQr and float(diag_length) < float(h+w)/5.0:
                # draw triangle!
                cv2.line(img, tuple(all_centers[i[0]]), tuple(all_centers[i[1]]), (255, 0, 255), 4)
                cv2.line(img, tuple(all_centers[i[1]]), tuple(all_centers[i[2]]), (255, 0, 255), 4)
                cv2.line(img, tuple(all_centers[i[2]]), tuple(all_centers[i[0]]), (255, 0, 255), 4)
                # making remap matrix
                cc_x = int( float(all_centers[i[0]][0]+all_centers[i[0]][0]+all_centers[i[0]][0])/3.0 )
                cc_y = int( float(all_centers[i[0]][1]+all_centers[i[0]][1]+all_centers[i[0]][1])/3.0 )

                sMx, dMx = FF.FF.getWarpMx(all_squires[i[0]], all_squires[i[1]], all_squires[i[2]], [cc_y, cc_x], corner_num)
                # thresh2 = 255-thresh
                rQr = FF.FF.getRightQr(sMx, dMx, gray)

                # saving QR
                cv2.imwrite(tmpDir + os.path.sep + 'QR_'+file, rQr)

                # QR recognition
                print( FF.FF.getQR(os.path.dirname(__file__)+os.path.sep + 'QR_'+file), file )


        cv2.imwrite(tmpDir + os.path.sep + file, img)
        cv2.imwrite(tmpDir + os.path.sep + 'bin_'+file, thresh)
