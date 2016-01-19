import numpy as np
import cv2, os, subprocess
from FF import FF

FF.FF.readSettings()
file = '1.2.png'
tmpDir = 'tmp'
abs_file_path = os.path.dirname(__file__)+'/balls2/'+file

img_origin = cv2.imread(abs_file_path)

black_cropped = FF.FF.black_and_white(img_origin.copy())
new_img = FF.FF.toSphere(black_cropped.copy())
cv2.imwrite(tmpDir + os.path.sep + file, new_img)

# rotate and apply distortion
h_res, w_res = black_cropped.shape
print(h_res, w_res)
M = cv2.getRotationMatrix2D((w_res/2, h_res/2), 45, 1)
black_rotated = cv2.warpAffine(black_cropped.copy(), M, (w_res, h_res))
new_img_rotated = FF.FF.toSphere(black_rotated.copy())
cv2.imwrite(tmpDir + os.path.sep + 'rot_'+file, new_img)

qr_found = FF.FF.getQR(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + file)
qr_init = FF.FF.getQR(abs_file_path)
qr_rotated = FF.FF.getQR(os.path.dirname(__file__)+os.path.sep+tmpDir+os.path.sep + 'rot_' + file)

if qr_init is not None:
    print('INIT FOUND '+qr_init)
if qr_found is not None:
    print('SPHERE NORM FOUND ' + qr_found )
if qr_rotated is not None:
    print('SPHERE ROTATED FOUND ' + qr_rotated )


# cv2.imshow("Image origin", img_origin)
cv2.imshow("Image origin", cv2.pyrDown(black_cropped))
cv2.imshow("Image rotated", cv2.pyrDown(new_img_rotated))
cv2.imshow("Image res", cv2.pyrDown(new_img))
cv2.waitKey(0)