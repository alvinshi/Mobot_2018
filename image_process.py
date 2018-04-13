import numpy as np
import cv2
import copy
import math
import os

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel < thresh[0]) | (channel> thresh[1])] = 1
    return binary_output

def rgb_select(img,thresh=(0,255)):
    ch=img[:,:,2]
    binary_output=np.zeros_like(ch)
    binary_output[(ch>thresh[0])&(ch<thresh[1])]=1
    return binary_output

# get the middle part of the image for image processing
def get_middle(img):
    rowNum = img.shape[0]
    colNum = img.shape[1]
    rowInterval = rowNum/4
    colInterval = colNum/4
    midRow = rowNum/2
    midCol = colNum/2
    # Take the middle one third of the image
    croppedImg = img[0:rowNum, midCol-colInterval:midCol+colInterval]
    return croppedImg

def thresholding(img):
    #x_thresh = sobel_thresh(img, orient='x', min=30,max=150)
    #mag_thr = mag_thresh(img, sobel_kernel=3, mag_thresh=(40, 150))
    #dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.8, 1.2))
    #rgb_thresh = rgb_select(img,(210,220))
    hls_thresh = hls_select(img, thresh=(40, 60))
    lab_thresh = lab_select(img, thresh=(140, 220))
    luv_thresh = luv_select(img, thresh=(170, 255))
    thresholded = np.zeros_like(hls_thresh)
    thresholded[((hls_thresh == 1) & (lab_thresh == 1)) & (luv_thresh==1)]=255
    return thresholded

# adaptive thresholding method
def adaptive_thresholding(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adaptive_threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    cv2.imshow("adaptive", adaptive_threshed)
    return adaptive_threshed

def row_segment_centor(img, NUM_SEGS):
    # Segment the original image into 20 segments
    numSegs = NUM_SEGS
    numRows = img.shape[0]
    numCols = img.shape[1]
    rowInterval = numRows/numSegs
    segmentCentors = [None] * numSegs

    startRow = 0
    for i in range(0, numSegs):
        imgSeg = img[startRow:startRow+rowInterval, 0:numCols]
        # Threshold imageSegments and calculate the centor of each segments
        imgSegThreshed = thresholding(imgSeg)
        coor = np.argwhere(imgSegThreshed == 255)
        if len(coor) == 0:
            rmean = img.shape[0]/2
            cmean = img.shape[1]/2
        else:
            rmean=int(math.floor(np.mean(coor[:,0])))
            cmean=int(math.floor(np.mean(coor[:,1])))
        segmentCentors[i] = (cmean, startRow+rmean)
        startRow = startRow + rowInterval   # upddate row
        label = "imgSeg %d" %i
    return segmentCentors

def decide_way(img):
    img=cv2.GaussianBlur(img,(25,25),0)
    blur=thresholding(img)
    coor=np.argwhere(blur==255)
    if len(coor) == 0:
        rmean = img.shape[0]/2
        cmean = img.shape[1]/2
    else:
        rmean=int(math.floor(np.mean(coor[:,0])))
        cmean=int(math.floor(np.mean(coor[:,1])))
    col=img.shape[1]/2
    if(cmean<col-30):
        command='Left'
    elif(cmean>col+30):
        command='Right'
    else:
        command='Straight'
    cv2.putText(img,command, (10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.rectangle(img,(cmean-20,rmean-20),(cmean+20,rmean+20),(0,255,0),3)
    return command,img

def main():
    PICTURE_FILE = './sample_pictures/WechatIMG78.jpeg'
    NUM_SEGS = 40

    img = cv2.imread(PICTURE_FILE)
    img = cv2.GaussianBlur(img,(13,13),0)
    img = get_middle(img)
    segmentCentors = row_segment_centor(img, NUM_SEGS)
    for i in range(0, NUM_SEGS):
        cv2.circle(img, segmentCentors[i], 5, (255,0,0))
    #command, img = decide_way(img)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
