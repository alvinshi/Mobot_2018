import numpy as np
import cv2
import copy
import math
import os
import time
import Threshold as th

MODE = "video"  # image or video
NUM_SEGS = 20   # Number of row slices
IMG_FRACTION = 0.75  # fraction of the middle image

# get the middle part of the image for image processing
def get_middle(img, fraction = 0.5):
    rowNum = img.shape[0]
    colNum = img.shape[1]
    colInterval = int(colNum*fraction/2)
    colOffset = int(colNum*(1-fraction)/2)
    midRow = rowNum/2
    midCol = colNum/2
    # Take the middle one third of the image
    croppedImg = img[0:rowNum, midCol-colInterval:midCol+colInterval]
    return croppedImg, colOffset

# Returns   1. Segment centors (including two different paths)
#           2. bool whether it is at intersection state
def row_segment_centers(img, NUM_SEGS, colOffset, CONNECTIVITY=8, AREA_THRESH=800):
    numSegs = NUM_SEGS
    numRows = img.shape[0]
    numCols = img.shape[1]
    rowInterval = numRows/numSegs
    startRow = 0
    centroids = []
    consecDiverge = 0
    atIntersection = False
    for i in range(0, numSegs):
        imgSeg = img[startRow:startRow+rowInterval, 0:numCols]
        output = cv2.connectedComponentsWithStats(imgSeg, CONNECTIVITY, cv2.CV_32S)
        labelNum = output[0]
        # check intersection
        if (i < numSegs/2):
            if labelNum == 2:
                consecDiverge += 1
            else:
                consecDiverge = 0
            if consecDiverge >= 4:
                atIntersection = True
            else:
                atIntersection = False

        # get centroids
        stats = output[2]
        for j in range(1, labelNum):    # Start from 1 to ignore background label
            if stats[j, cv2.CC_STAT_AREA] > AREA_THRESH:
                x = int(output[3][j][0])+colOffset
                y = int(output[3][j][1])+startRow
                centroids.append((x, y))
        startRow += rowInterval

    if (atIntersection):
        print("at intersection")
    else:
        print("not at intersection")

    return centroids

def image_process(img, NUM_SEGS, IMG_FRACTION):
    img = cv2.GaussianBlur(img,(13,13),0)
    imgThreshed = th.thresholding(img)
    imgMiddle, colOffset = get_middle(imgThreshed, IMG_FRACTION)
    centroids = row_segment_centers(imgMiddle, NUM_SEGS, colOffset)

    for i in range(0,len(centroids)):
        cv2.circle(img, centroids[i], 5, (255,0,0))

    cv2.imshow('imageThreshed',imgThreshed)
    cv2.imshow('image', img)

def main():

    PICTURE_FILE = './sample_pictures/200.jpg'
    VIDEO_FILE = './videos/output.avi'
    global NUM_SEGS
    global IMG_FRACTION

    if MODE == "image":
        img = cv2.imread(PICTURE_FILE)
        image_process(img, NUM_SEGS, IMG_FRACTION)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        vid = cv2.VideoCapture(VIDEO_FILE)
        while True:
            ret, frame = vid.read()
            image_process(frame, NUM_SEGS, IMG_FRACTION)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break


if __name__ == '__main__':
    main()
