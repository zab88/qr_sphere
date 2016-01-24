import cv2, os
from scipy.interpolate import griddata
import numpy as np
from FF import FF

img_origin = cv2.imread('main.png')
img_grey = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
img_grey = cv2.blur(img_grey, (3,3))

cv2.circle(img_origin, (55, 43), 2, (0, 0, 255), 2)
cv2.circle(img_origin, (50, 282), 2, (0, 0, 255), 2)
cv2.circle(img_origin, (285, 274), 2, (0, 0, 255), 2)
cv2.circle(img_origin, (280, 36), 2, (0, 0, 255), 2)

cv2.circle(img_origin, (170, 40), 2, (0, 255, 0), 2)
cv2.circle(img_origin, (46, 166), 2, (0, 255, 0), 2)
cv2.circle(img_origin, (280, 160), 2, (0, 255, 0), 2)
cv2.circle(img_origin, (168, 280), 2, (0, 255, 0), 2)

cv2.circle(img_origin, (163, 168), 2, (255, 0, 0), 2)

grid_x, grid_y = np.mgrid[0:199:200j, 0:199:200j]
destination = np.array([[0,0], [0,99], [0,199],
                  [99,0],[99,99],[99,199],
                  [199,0],[199,99],[199,199]])
# source = np.array([[55, 43], [46,170], [50,282],
#                   [170,40],[163,168],[168,280],
#                   [280,36],[280,160],[285,274]])
source = np.array([[43, 55], [40,170], [36,280],
                  [166,46],[168,163],[160,280],
                  [282,50],[280,168],[274,285]])

grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(200,200)
map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(200,200)
map_x_32 = map_x.astype('float32')
map_y_32 = map_y.astype('float32')

warped = cv2.remap(img_grey, map_x_32, map_y_32, cv2.INTER_CUBIC)#cv2.INTER_LINEAR
warped2 = cv2.threshold(cv2.blur(warped, (3, 3)), 127, 255, cv2.THRESH_OTSU)[1]

warped = cv2.copyMakeBorder(warped, 40, 40, 40, 40, cv2.BORDER_CONSTANT, warped, (255, 255, 255))

cv2.imshow("Image origni", img_origin)
cv2.imshow("Image grey", img_grey)
cv2.imshow("Image remapped", warped)
cv2.imshow("WARPED 2", warped2)
cv2.imwrite("warped.png", warped)
cv2.imwrite("warped2.png", warped2)


print( FF.FF.getQR(os.path.dirname(__file__)+os.path.sep + "warped.png") )
cv2.waitKey(0)
cv2.destroyAllWindows()






    # if len(approx)==5:
    #     print "pentagon"
    #     cv2.drawContours(img,[cnt],0,255,-1)
    # elif len(approx)==3:
    #     print "triangle"
    #     cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    # elif len(approx)==4:
    #     print "square"
    #     cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    # elif len(approx) == 9:
    #     print "half-circle"
    #     cv2.drawContours(img,[cnt],0,(255,255,0),-1)
    # elif len(approx) > 15:
    #     print "circle"
    #     cv2.drawContours(img,[cnt],0,(0,255,255),-1)



###!  np.mgrid[-1:1:5j] array([-1. , -0.5,  0. ,  0.5,  1. ])
# grid_x, grid_y = np.mgrid[0:149:150j, 0:149:150j]
# destination = np.array([[0,0], [0,49], [0,99], [0,149],
#                   [49,0],[49,49],[49,99],[49,149],
#                   [99,0],[99,49],[99,99],[99,149],
#                   [149,0],[149,49],[149,99],[149,149]])
# source = np.array([[22,22], [24,68], [26,116], [25,162],
#                   [64,19],[65,64],[65,114],[64,159],
#                   [107,16],[108,62],[108,111],[107,157],
#                   [151,11],[151,58],[151,107],[151,156]])
# grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
# map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(150,150)
# map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(150,150)
# map_x_32 = map_x.astype('float32')
# map_y_32 = map_y.astype('float32')
#
# orig = cv2.imread("main.png")
# warped = cv2.remap(orig, map_x_32, map_y_32, cv2.INTER_CUBIC)
# # cv2.imwrite("warped.png", warped)
# cv2.imshow("Image remapped", warped)
# cv2.waitKey(0)