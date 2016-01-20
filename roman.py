import numpy as np
import cv2
import os
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt

def crop_qr(squares_dict, image):
    #find three corner points of QR code
    corners = find_corners(squares_dict)
    # find 4th point and construct QR code border
    full_corners = cv2.convexHull(get_full_corners(corners).astype(np.int32))
    
    # enlarge QR code border a bit to fit skewed codes
    full_corners = scale_bound(full_corners.squeeze(), 1.2)
    
    pts1 = full_corners.astype(np.float32)
    pts2 = np.float32([[150,150],[0,150],[0,0],[150,0]])
    
    # do linear transformation from 4 enlarged border points to a square
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image,M,(150,150))
    
    # sharpen image using reversed medianBlur
    dst = cv2.addWeighted(dst, 1.5, cv2.medianBlur(dst,5), -0.4, 0)
    return dst, full_corners

def distort_image(src, frac):
# this function does barrel distortion correction with openCV camera lens correction tools
# distortion coefficients was found manually for better sphere rectification so it can be changed
    width  = src.shape[1]
    height = src.shape[0]
    distCoeff = np.zeros((5,1),np.float64)

    distCoeff[0,0] = -frac
    distCoeff[1,0] = frac / 2
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0 
    cam[1,2] = height/2.0 
    cam[0,0] = 400.        
    cam[1,1] = cam[0,0]    

    cam_new, roi=cv2.getOptimalNewCameraMatrix(cam, distCoeff, (width,height), 1)
    dst = cv2.undistort(src, cam, distCoeffs=distCoeff, newCameraMatrix=cam_new)
    return dst


def find_corners(cur_dict):
#     function finds three QR code absolute corners from three squares
    corners = np.zeros((3, 2))
    mass_center = np.vstack(list(cur_dict.values())).squeeze().mean(axis=0, keepdims=True)
    # set points which has max distance to triangle mass center in each position square as absolute corners
    for ind, square in enumerate(list(cur_dict.values())):
        corner_points = square.squeeze()
        distances = cdist(corner_points, mass_center)
        max_three = distances.squeeze().argmax()
        corners[ind, :] = corner_points[max_three]
    return corners

def get_qr_squares(squares):
#     function finds two potentional QR codes
    centroids = np.array([square.mean(axis=0) for square in squares])
    first_qr = {}
    second_qr = {}
    min_rad = np.inf
    min_rad_2 = np.inf
    first_qr['A'], first_qr['B'], first_qr['C'] = 0, 0, 0
    # find nearest triples of position squares
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            for k in range(j+1, len(centroids)):
                # find sum of pairwise large squares distances
                rad = cv2.norm(centroids[i], centroids[j]) + cv2.norm(centroids[i], centroids[k]) + cv2.norm(centroids[k], centroids[j])
                if rad < min_rad:
                    min_rad_2 = min_rad
                    second_qr['A'], second_qr['B'], second_qr['C'] = first_qr['A'], first_qr['B'], first_qr['C']
                    min_rad = rad
                    first_qr['A'], first_qr['B'], first_qr['C'] = squares[i], squares[j], squares[k]
    
    is_2_exists = np.isinf(min_rad_2)
    return first_qr, second_qr, is_2_exists

def get_full_corners(corners):
#     function finds 4th corner point from three corners
    #compute triangle sides lengths 
    dist_matrix = squareform(pdist(corners))
    #find sharp and 90 corners points
    a_sharp, b_sharp = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    c_square = ({0,1,2} - {a_sharp} - {b_sharp}).pop()
#     find 4th point from second qr square diagonal
    diag_middle = (corners[a_sharp] + corners[b_sharp]) / 2.
    v = corners[c_square] - diag_middle
    u = v / cv2.norm(v)
    d_square = diag_middle - cv2.norm(v)*u
    return np.vstack([corners, d_square])

def scale_bound(corners, frac):
#     enlarge QR bound by factor
    mass_center = corners.mean(axis=0)
    new_corners = np.zeros(corners.shape)
    for i in range(len(corners)):
        v = corners[i] - mass_center
        u = v / cv2.norm(v)
        new_corners[i, :] = mass_center + frac*cv2.norm(v)*u    
    return new_corners
	
def find_qr_code(image):
    #find qr position squares contours using canny
    gray = cv2.GaussianBlur(image, (3, 3), 0)
    closed = cv2.Canny(gray, 200, 10)
    _, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    
    # show borders by canny
#     cv2.imshow('image', closed)
#     cv2.waitKey(0)
    
    # loop over the contours
    squares = []
    for c in cnts:
    # approximate the contour (connect contours with small breaks)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)

    # detect squares by 4 points and area ratios
        sq_ratio = cv2.contourArea(approx) / (gray.shape[0]*gray.shape[1]) * 1e6
        if len(approx) == 4 and sq_ratio > 1500 and sq_ratio < 10000:
            # add green positional squares borders on image
#             cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
            total += 1
            squares.append(approx)
    # minimum 3 recognized position squares needed to find a QR code
    if len(squares) < 3:
        return
    
    # find two potential QR codes
    first, second, is_2_exists = get_qr_squares(squares)
    
    # crop first QR
    qr_1, full_corners_1 = crop_qr(first, image)
    
    #add QR code border on image
    cv2.drawContours(image, [full_corners_1.astype(np.int32)], -1, (0, 0, 255), 2)
    
    #show cropped QR code
    cv2.imshow('image', qr_1)
    cv2.waitKey(0)

    if not is_2_exists:
        qr_2, full_corners_2 = crop_qr(second, image)
        
        #add QR code border on image
        cv2.drawContours(image, [full_corners_2.astype(np.int32)], -1, (0, 0, 255), 2)

        #show cropped QR code
        cv2.imshow('image', qr_2)
        cv2.waitKey(0)
        
    
    #show skewed image
    cv2.imshow('image', image)
    cv2.waitKey(0)
	
def find_sphere(image_orig):
    image = image_orig[36:-55, 45:-75].copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, image.shape[0], minRadius = int(image.shape[0]/2*0.95),
                               maxRadius = int(image.shape[0]/2*1.05), param1 = 10, param2 = 100)
     
    # ensure at least some circles were found
    if circles is not None: 
        circles = np.round(circles[0, :]).astype("int")

        (x, y, r) = circles[0]
        image = image[max(y-r-10, 0):min(y+r+10, image.shape[1]), max(x-r-10, 0):min(x+r+10, image.shape[1])]

    else:
        image = image_orig[38:-57, 72:-94]
    
    image = cv2.resize(image, (340, 340), interpolation = cv2.INTER_CUBIC)
    return image

	
	
if __name__ == "__main__":
    for i in os.listdir():
        image_orig = cv2.imread(i)
        image = find_sphere(image_orig)
        for frac in np.linspace(0.8, 1.1, 10):
            image_distorted = distort_image(image, frac)
            find_qr_code(image_distorted)