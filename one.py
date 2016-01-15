import numpy as np
import cv2, os, subprocess
from FF import FF

file = '51.jpg'
tmpDir = 'tmp'
abs_file_path = os.path.dirname(__file__)+'/balls_old/'+file

img_origin = cv2.imread(abs_file_path)

new_img = FF.FF.black_and_white(img_origin.copy())
new_img = FF.FF.toSphere(new_img)
cv2.imwrite(tmpDir + os.path.sep + file, new_img)

print(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + file)
qr_found = FF.FF.getQR(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + file)
if qr_found is not None:
    print qr_found
else:
    qr_init = FF.FF.getQR(abs_file_path)
    if qr_init is not None:
        print('INIT FOUND '+qr_init)

cv2.imshow("Image origin", img_origin)
cv2.imshow("Image", new_img)
cv2.waitKey(0)