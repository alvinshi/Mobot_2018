import numpy as np
import cv2
import copy
import math
import os
import time

doubleRasterTime = 0

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

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hchannel = hls[:,:,0]
    lchannel=hls[:,:,1]
    schannel=hls[:,:,2]
    binary_output = np.zeros_like(lchannel)
    binary_output[(lchannel > thresh[0]) & (lchannel< thresh[1]) & (schannel < 100)] = 1
    return binary_output

def rgb_select(img,thresh=(0,255)):
    ch0=img[:,:,0]
    ch1=img[:,:,1]
    ch2=img[:,:,2]
    binary_output=np.zeros_like(ch0)
    binary_output[(ch0>thresh[0])&(ch0<thresh[1])&(ch1>thresh[0])&(ch1<thresh[1])&(ch2>thresh[0])&(ch2<thresh[1])]=1
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

# Dilation to expand white line after thresholding
def dilation(img):
    kernel = np.ones((17,17), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    return img_dilation

def thresholding(img):
    # x_thresh = sobel_thresh(img, orient='x', min=30,max=150)
    # mag_thr = mag_thresh(img, sobel_kernel=3, mag_thresh=(40, 150))
    # dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.8, 1.2))
    # rgb_thresh = rgb_select(img,(210,220))
    lab_thresh = lab_select(img, thresh=(140, 220))
    luv_thresh = luv_select(img, thresh=(170, 255))
    hls_thresh = hls_select(img, thresh=(100, 255))
    rgb_thresh = rgb_select(img, thresh=(150, 190))
    thresholded = np.zeros_like(hls_thresh)
    thresholded[(hls_thresh == 1)]=255
    return dilation(thresholded)

# adaptive thresholding method
def adaptive_thresholding(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adaptive_threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    cv2.imshow("adaptive", adaptive_threshed)
    return adaptive_threshed

# Normalize the thresholded image to a binary image
def normalize(img):
    normalizeImg = np.zeros_like(img)
    normalizeImg[img == 255] = 1
    return normalizeImg

def get_center(coordinates):
    index = 0
    centers = []
    while coordinates[index] != None:
        sums = [0,0]
        for i in coordinates[index]:
            (row, col) = i
            sums[0] = sums[0] + row
            sums[1] = sums[1] + col
        sums[0] = int(math.floor(sums[0] / len(coordinates[index])))
        sums[1] = int(math.floor(sums[1] / len(coordinates[index])))
        centers.append(tuple(sums))
        index = index + 1
    return centers

# Switch the row and col for the drawing function
def switchRowCol(origCoor):
    col = origCoor[1]
    row = origCoor[0]
    return [col, row]

# add the coordinates of same label to "coordinates"
def addCoordinates(coorNum, label, labelCoor, startRow, coordinates):
    for index in range(0, coorNum):
        labelCoor[index][0] = labelCoor[index][0] + startRow
        labelCoorT = switchRowCol(labelCoor[index])
        if coordinates[label-2] != None:
            coordinates[label-2].append(labelCoorT)
        else:
            coordinates[label-2] = [labelCoorT]

# double raster for image segmentation
# returns the center coordinates of each of the segment
def double_raster(imgTakein, startRow):
    # take in binary image; startRow is the start row of the current image slice
    img = normalize(imgTakein)
    cur_label=2
    coordinates = [None] * 50
    eq=[0] * len(img)*len(img[0])
    for row in range(0,len(img)):
        for col in range(0,len(img[row])):
            if(img[row][col]==1):
                if(row>0):
                    up=img[row-1][col]
                else:
                    up=0
                if(col>0):
                    left=img[row,col-1]
                else:
                    left=0
                if(up==0 and left==0):
                    img[row][col]=cur_label
                    cur_label=cur_label+1
                elif(up!=0 and left!=0):
                    img[row][col]=min(up,left)
                    if(up!=left):
                        eq[max(up,left)]=min(up,left)
                        a=min(up,left)
                        while(eq[a]!=0):
                            eq[max(up,left)]=eq[a]
                            a=eq[a]
                elif(up==0 or left==0):
                    img[row][col]=max(up,left)

    # changed nested for loop of the second sweep to below, faster for 5-6 second
    max_label = cur_label   # record the max label number
    labelPixNumber = [0] * max_label    # The number of pixels in each label
    coorAdded = False   # switch of whether the coordinates has been recorded
    for label in range(0, max_label):
        labelCoor = np.argwhere(img == label)   # get the coordinates of pixels with same label
        coorNum = len(labelCoor)
        labelPixNumber[label] = coorNum
        if (eq[label] != 0):
            eqLabel = eq[label]
            img = eqLabel * (img == label) + img
            # Add the number of pixels of the current label to the equiv label
            # and set the current label pixel number to 0
            labelPixNumber[eqLabel] = labelPixNumber[eqLabel] + labelPixNumber[label]
            labelPixNumber[label] = 0
            addCoordinates(coorNum, eqLabel, labelCoor, startRow, coordinates)
            coorAdded = True
        if not coorAdded:
            addCoordinates(coorNum, label, labelCoor, startRow, coordinates)
        coorAdded = False

    centers = get_center(coordinates)
    # print("finished double raster for one slice of image")
    return centers

# Returns   1. Segment centors (including two different paths)
#           2. bool path diverge state
def row_segment_center(img, NUM_SEGS):
    global doubleRasterTime
    # Segment the original image into 20 segments
    numSegs = NUM_SEGS
    numRows = img.shape[0]
    numCols = img.shape[1]
    rowInterval = numRows/numSegs
    segmentCentors = [None] * numSegs
    blockCenters = []
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
        startRow = startRow + rowInterval   # update row
        doubleRasterStart = time.time()
        blockCenters.append(double_raster(imgSegThreshed, startRow))
        doubleRasterEnd = time.time()

        doubleRasterTime += doubleRasterEnd - doubleRasterStart

    return segmentCentors, blockCenters

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
    startRunTime = time.time()
    PICTURE_FILE = './sample_pictures/100.jpg'
    NUM_SEGS = 40

    img = cv2.imread(PICTURE_FILE)
    cv2.imshow("thresholded", thresholding(img))
    img = cv2.GaussianBlur(img,(13,13),0)
    img = get_middle(img)
    segmentCentors, blockCenters = row_segment_center(img, NUM_SEGS)

    for i in range(0, NUM_SEGS):
        cv2.circle(img, segmentCentors[i], 5, (255,0,0))
        for j in range(0, len(blockCenters[i])):
            cv2.circle(img, blockCenters[i][j], 5, (0,0,255))

    endRunTime = time.time()
    runTime = endRunTime - startRunTime
    segmentTime = endSegmentTime - startSegmentTime
    print("Total run time: %f" %runTime)
    print("Total segment time: %f" %segmentTime)
    print("DoubleRaster time: %f" %doubleRasterTime)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
