import cv2
import ConfigParser

Config = ConfigParser.ConfigParser()
Config.read("settings.ini")

for i in range(1, 4, 1):
    crop_top = Config.getint('CAM_'+str(i), 'crop_top')
    crop_left = Config.getint('CAM_'+str(i), 'crop_left')
    crop_bottom = Config.getint('CAM_'+str(i), 'crop_bottom')
    crop_right = Config.getint('CAM_'+str(i), 'crop_right')

    img = cv2.imread('balls_gold/1.1.'+str(i)+'.png')
    h, w = img.shape[:2]
    cv2.rectangle(img, (crop_left, crop_top), (w-crop_right, h-crop_bottom), (0, 0, 255), 3)

    cv2.imshow('CAM_'+str(i), cv2.pyrDown(img.copy()))

    print(w-crop_left-crop_right, h-crop_top-crop_bottom)
cv2.waitKey(0)
