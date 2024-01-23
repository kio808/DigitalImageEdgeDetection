import cv2
import numpy as np

# For recording time
import time

# For calculating PSNR
from math import log10, sqrt

# Used for SSIM
from skimage import metrics

import os
import random

# IF STUCK ON PICTURES/RESULTS, PRESS ANY KEY TO CONTINUE THE CODE

# Determines the range of gray RGB values
MinGray = (255,255,255)
MaxGray = (105,105,105)

# Picks random image from test folder
random_img = random.choice(os.listdir('test'))
print(random_img)

random_road = cv2.imread(os.path.join('test', random_img))
cv2.imshow(str(random_img), random_road)

# Finds the expected roads for the random picture to compare
correct_road = cv2.imread(os.path.join('test_labels', (random_img.replace('tiff', 'tif'))))
cv2.imshow(str(random_img.replace('tiff', 'tif')), correct_road)

img = cv2.imread(os.path.join('test', random_img))
cv2.waitKey(1)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img, (3,3), 0)

# Creates mask to only keep pixels that are in this range
gray_mask = cv2.inRange(img, MaxGray, MinGray)

# Uses mask to keep pixels that are in this range to the image
result = cv2.bitwise_and(img_blur, img_blur, mask=gray_mask)
cv2.imshow("RESULT", result)

# Erodes the image
result_erosion = cv2.erode(gray_mask, (5,5), iterations=3)
cv2.imshow("EROSION", result_erosion)

# Dilates the image
open = cv2.dilate(result_erosion, (5,5), iterations=3)
cv2.imshow('DILATION', open)

# Erodes the image
erode2 = cv2.erode(open, (5,5), iterations=3)
cv2.imshow('ERODE2', erode2)

# Adds a median filter to the image
median = cv2.medianBlur(erode2, 3)
cv2.imshow('MEDIAN', median)

cv2.waitKey(0)

# Hough Transform
def Hough(blackbg):
    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(erode2,1, np.pi/180,100, minLineLength, maxLineGap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blackbg,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('LINES', blackbg)
    cv2.waitKey(1)

    cv2.imwrite("output.jpg", blackbg)
    cv2.waitKey(1)

    output = cv2.imread('output.jpg')
    return output

#Prewitt
def Prewitt():
    Prewitt_time = time.time()

    prewittbg = cv2.imread("PrewittBlack.png")

    prewittx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewitty = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(median, -1, prewittx)
    img_prewittxy = cv2.filter2D(img_prewittx, -1, prewitty)

    cv2.imshow('Prewitt', img_prewittxy)
    cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(img_prewittxy,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(prewittbg, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Contours', prewittbg)
    cv2.waitKey(1)

    hough = Hough(prewittbg)
    print("PREWITT:")
    PSNR(hough)
    SSIM(hough)

    print("Time: %s" % (time.time() - Prewitt_time))

    cv2.imwrite("PrewittOutput.png", prewittbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Canny Edge Detection
def CannyEdge():
    Canny_time = time.time()
    cannybg = cv2.imread('CannyBlack.png')
    edges = cv2.Canny(image=median, threshold1=100, threshold2=200)

    contours, hierarchy = cv2.findContours(edges,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cannybg, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Contours', cannybg)
    cv2.waitKey(1)

    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(1)
    hough = Hough(cannybg)
    print("CANNY EDGE:")
    PSNR(hough)
    SSIM(hough)

    print("Time: %s" % (time.time() - Canny_time))

    cv2.imwrite("CannyOutput.png", cannybg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Sobel Edge Detection
def Sobel():

    Sobel_time = time.time()

    sobelbg = cv2.imread('SobelBlack.png')
    MySobelX = np.array ([[-1,-2,-1], [0,0,0], [1,2,1]])
    MySobelY = np.array ([[-1,0,1], [-2,0,2], [-1,0,1]])

    myx = cv2.filter2D(median, -1, MySobelX)
    myxy = cv2.filter2D(myx, -1, MySobelY)

    cv2.imshow('Sobel with Gaussian', myxy)
    cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(myxy,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(sobelbg, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Contours', sobelbg)
    cv2.waitKey(1)

    hough = Hough(sobelbg)
    print("SOBEL:")
    PSNR(hough)
    SSIM(hough)

    print("Time: %s" % (time.time() - Sobel_time))

    cv2.imwrite("SobelOutput.png", sobelbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#ROBERTS CROSS
def Robert():
    Robert_time = time.time()

    robertbg = cv2.imread("RobertBlack.png")

    RobertX = np.array([[1,0], [0,-1]])
    RobertY = np.array([[0,1], [-1,0]])

    RobCrossX = cv2.filter2D(median, -1, RobertX)
    RobCrossXY = cv2.filter2D(RobCrossX, -1, RobertY)

    cv2.imshow('Robert Cross with Gaussian', RobCrossXY)
    cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(RobCrossXY,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(robertbg, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Contours', robertbg)
    cv2.waitKey(1)

    hough = Hough(robertbg)
    print("ROBERTS CROSS:")
    PSNR(hough)
    SSIM(hough)

    print("Time: %s" % (time.time() - Robert_time))

    cv2.imwrite("RobertOutput.png", robertbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Laplacian():
    Laplacian_time = time.time()

    laplacianbg = cv2.imread("LaplacianBlack.png")

    LaplacianHeavy = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    LH = cv2.filter2D(median, -1, LaplacianHeavy)

    cv2.imshow('Laplacian Heavy', LH)
    cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(LH,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(laplacianbg, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Contours', laplacianbg)
    cv2.waitKey(1)

    hough = Hough(laplacianbg)
    print("LAPLACIAN:")
    PSNR(hough)
    SSIM(hough)

    print("Time: %s" % (time.time() - Laplacian_time))

    cv2.imwrite("LaplacianOutput.png", laplacianbg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




# PSNR
def PSNR(output):
    mse = np.mean((correct_road - output) ** 2)
    if(mse == 0):
        mse = 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))

    print("MSE IS "  + str(mse))
    print("PSNR IS " + str(psnr))

# SSIM
def SSIM(output):
    correct_grayscale = cv2.cvtColor(correct_road, cv2.COLOR_BGR2GRAY)
    output_grayscale = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ssim_score = metrics.structural_similarity(correct_grayscale, output_grayscale, full=True)
    print("SSIM IS " + str(round(ssim_score[0], 2)))


CannyEdge()
Sobel()
Prewitt()
Robert()
Laplacian()
cv2.destroyAllWindows()

LO = cv2.imread("LaplacianOutput.png")
SO = cv2.imread("SobelOutput.png")
PO = cv2.imread("PrewittOutput.png")
RO = cv2.imread("RobertOutput.png")
CO = cv2.imread("CannyOutput.png")

cv2.imshow("LaplacianOutput", LO)
cv2.imshow("SobelOutput", SO)
cv2.imshow("PrewittOutput", PO)
cv2.imshow("RobertOutput", RO)
cv2.imshow("CannyOutput", CO)
cv2.imshow(str(random_img.replace('tiff', 'tif')), correct_road)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print("--- %s seconds ---" % (time.time() - start_time))

