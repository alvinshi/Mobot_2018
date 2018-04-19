import numpy as np
import cv2
import copy
import math
import os
import time
import Threshold as th

MODE = "video"  # image or video
NUM_SEGS = 35  # Number of row slices
IMG_FRACTION = 0.75  # fraction of the middle image
PICTURE_FILE = './sample_pictures/200.jpg'
VIDEO_FILE = './videos/output.avi'

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
def row_segment_centers(img, NUM_SEGS, colOffset, CONNECTIVITY=8, AREA_THRESH=500):
    numSegs = NUM_SEGS
    numRows = img.shape[0]
    numCols = img.shape[1]
    rowInterval = numRows/numSegs
    startRow = 0
    midCentroids = []   # centroids add to here when not at intersection
    leftCentroids = []  # left centroids add to here when at intersection
    rightCentroids = [] # right centroids add to here when at intersection
    midStartIndex = 0
    consecDiverge = 0
    frameAtIntersection = False
    for i in range(0, numSegs):
        imgSeg = img[startRow:startRow+rowInterval, 0:numCols]
        output = cv2.connectedComponentsWithStats(imgSeg, CONNECTIVITY, cv2.CV_32S)
        labelNum = output[0]
        # check intersection
        if i < numSegs/2:
            if labelNum == 3:
                consecDiverge += 1
            else:
                consecDiverge = 0
            if consecDiverge >= 10:
                frameAtIntersection = True
        # get centroids
        stats = output[2]
        midCenterPerSlice = 0
        for j in range(1, labelNum):    # Start from 1 to ignore background label
            if stats[j, cv2.CC_STAT_AREA] > AREA_THRESH:
                x = int(output[3][j][0])+colOffset
                y = int(output[3][j][1])+startRow
                midCentroids.append((x, y))
                midCenterPerSlice += 1
                if labelNum == 2:   # One segment
                    leftCentroids.append((x,y))
                    rightCentroids.append((x,y))
        if midCenterPerSlice == 2:   #   two segments
            # Add left and right centroids, should be 2 in list
            x1 = midCentroids[midStartIndex][0]
            x2 = midCentroids[midStartIndex][0]
            if x1 < x2:
                leftCentroids.append((midCentroids[midStartIndex][0], midCentroids[midStartIndex][1]))
                rightCentroids.append((midCentroids[midStartIndex+1][0], midCentroids[midStartIndex+1][1]))
            else:
                leftCentroids.append((midCentroids[midStartIndex+1][0], midCentroids[midStartIndex+1][1]))
                rightCentroids.append((midCentroids[midStartIndex][0], midCentroids[midStartIndex][1]))
        midStartIndex += midCenterPerSlice
        startRow += rowInterval

    if not frameAtIntersection:
        leftCentroids = None
        rightCentroids = None

    return midCentroids, leftCentroids, rightCentroids, frameAtIntersection

# Return the collections of centriods
#        whether the image represents an intersection
def image_process(img, seq_n, NUM_SEGS, IMG_FRACTION):
    cv2.imwrite(str(seq_n) + "_in.jpg", img)
    img = cv2.GaussianBlur(img,(13,13),0)
    imgThreshed = th.thresholding(img)
    imgMiddle, colOffset = get_middle(imgThreshed, IMG_FRACTION)
    midCentroids, leftCentroids, rightCentroids, frameAtIntersection = row_segment_centers(imgMiddle, NUM_SEGS, colOffset)

    # Draw all
    for i in range(0,len(midCentroids)):
        cv2.circle(img, midCentroids[i], 5, (255,0,0))
    if leftCentroids != None and rightCentroids != None:
        for i in range(0,len(leftCentroids)):
            cv2.circle(img, leftCentroids[i], 3, (0,255,0))
        for i in range(0,len(rightCentroids)):
            cv2.circle(img, rightCentroids[i], 7, (0,0,255))
    # cv2.imshow('imageThreshed',imgThreshed)
    # cv2.imshow('image', img)
    cv2.imwrite(str(seq_n) + "_th.jpg", imgThreshed)
    cv2.imwrite(str(seq_n) + "_out.jpg", img)
    return midCentroids, leftCentroids, rightCentroids, frameAtIntersection

def get_commandInfo(imgCenter, centroids, STRAIGHT_TOL = 30):
    sumX = 0
    centroidNum = len(centroids)
    for centroid in centroids:
        sumX += centroid[0]
    if sumX/centroidNum < imgCenter[0]-STRAIGHT_TOL:
        command = "Left"
    elif sumX/centroidNum > imgCenter[0]+STRAIGHT_TOL:
        command = "Right"
    else:
        command = "Straight"
    return command


# Called in the robot main loop
# Takes in the image to be processed
# Takes in last N frames of intersection state (to get rid of noise)
# Takes in a list of choices, destructively modified the list
# The sequence number of the image past in
# return command and current "atIntersection" state
def get_command(img, pastStates, choices, seq_n, NUM_SEGS = NUM_SEGS, IMG_FRACTION = IMG_FRACTION):
    midCentroids, leftCentroids, rightCentroids, frameAtIntersection = image_process(img, seq_n, NUM_SEGS, IMG_FRACTION)
    width = img.shape[1]
    height = img.shape[0]
    imgCenter = (width/2, height/2)
    for i in range(0, len(pastStates)):
        if not pastStates[i]:
            command = get_commandInfo(imgCenter, midCentroids)
            return command, frameAtIntersection

    # Ready to make intersection decision
    preferredSide = choices.pop(0)
    if preferredSide == "left":
        command = get_commandInfo(imgCenter, leftCentroids)
        return command, frameAtIntersection
    elif preferredSide == "right":
        command = get_commandInfo(imgCenter, rightCentroids)
        return command, frameAtIntersection
    else:
        return None

def main():
    global PICTURE_FILE
    global VIDEO_FILE
    global NUM_SEGS
    global IMG_FRACTION

    if MODE == "image":
        img = cv2.imread(PICTURE_FILE)
        image_process(img, 0, NUM_SEGS, IMG_FRACTION)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        vid = cv2.VideoCapture(VIDEO_FILE)
        seq_n = 0
        while True:
            ret, frame = vid.read()
            image_process(frame, seqn_n, NUM_SEGS, IMG_FRACTION)
            seq_n += 1
            if cv2.waitKey(10) & 0xff == ord('q'):
                break


# if __name__ == '__main__':
#     main()
