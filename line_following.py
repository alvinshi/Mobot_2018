import numpy as np
import cv2
import copy
import math
import os
import time

def double_raster(imgTakein, startRow):
    # take in binary image; startRow is the start row of the current image slice
    img = normalize(imgTakein)
    cur_label=2
    coordinates = [None] * 50
    eq=[]
    for i in range(0,len(img)*len(img[0])):
        eq.append(0)
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
    #print("finished double raster for one slice of image")
    return centers

def sobel_thresh(img,orient='x',min=0,max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= min) & (scaled_sobel <= max)] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def luv_select(img, channel='l',thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    u_channel=luv[:,:,1]
    v_channel=luv[:,:,2]
    binary_output = np.zeros_like(l_channel)
    if(channel=='l'):
        binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    elif(channel=='u'):
        binary_output[(u_channel > thresh[0]) & (u_channel <= thresh[1])] = 1
    else:
        binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, channel='l',thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    a_channel=lab[:,:,1]
    b_channel=lab[:,:,2]
    binary_output = np.zeros_like(l_channel)
    if(channel=='l'):
        binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    elif(channel=='a'):
        binary_output[(a_channel > thresh[0]) & (a_channel <= thresh[1])] = 1
    else:
        binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

def hls_select(img,channel='l',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2] 
    binary_output = np.zeros_like(l_channel)
    if channel=='h':
        binary_output[(h_channel > thresh[0]) & (h_channel< thresh[1])] = 1
    elif channel=='l':
        binary_output[(l_channel > thresh[0]) & (l_channel< thresh[1])] = 1
    else:
        binary_output[(s_channel > thresh[0]) & (s_channel< thresh[1])] = 1
    return binary_output

def rgb_select(img,thresh=(0,255)):
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    binary_output=np.zeros_like(r)
    binary_output[(b>thresh[0])&(b<thresh[1])&(g>thresh[0])&(g<thresh[1])&(r>thresh[0])&(r<thresh[1])]=1
    return binary_output

def get_middle(img):
    rowNum = img.shape[0]
    colNum = img.shape[1]
    rowInterval = rowNum//4
    colInterval = colNum//4
    midRow = rowNum//2
    midCol = colNum//2
    # Take the middle one third of the image
    croppedImg = img[0:rowNum, midCol-colInterval:midCol+colInterval]
    return croppedImg

# Dilation to expand white line after thresholding
def dilation(img):
    kernel = np.ones((17,17), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    return img_dilation

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
        #print(label-2)
        if coordinates[label-2] != None:
            coordinates[label-2].append(labelCoorT)
        else:
            coordinates[label-2] = [labelCoorT]

def row_segment_centor(img, NUM_SEGS):
    # Segment the original image into 20 segments
    numSegs = NUM_SEGS
    numRows = img.shape[0]
    numCols = img.shape[1]
    rowInterval = numRows//numSegs
    segmentCentors = [None] * numSegs
    blockCenters = []
    startRow = 0
    for i in range(0, numSegs):
        imgSeg = img[startRow:startRow+rowInterval, 0:numCols]
        # Threshold imageSegments and calculate the centor of each segments
        imgSegThreshed = thresholding(imgSeg)
        coor = np.argwhere(imgSegThreshed == 255)
        if len(coor) == 0:
            rmean = img.shape[0]//2
            cmean = img.shape[1]//2
        else:
            rmean=int(math.floor(np.mean(coor[:,0])))
            cmean=int(math.floor(np.mean(coor[:,1])))

        segmentCentors[i] = (cmean, startRow+rmean)
        startRow = startRow + rowInterval   # update row
        blockCenters.append(double_raster(imgSegThreshed, startRow))

    return segmentCentors, blockCenters

def thresholding(img):
    rgb_thresh = rgb_select(img,(150,255))
    hls_thresh = hls_select(img,channel='l', thresh=(180,240 ))
    #lab_thresh = lab_select(img, channel='l',thresh=(190, 240))
    #luv_thresh = luv_select(img, channel='l',thresh=(180, 240))
    threshholded = np.zeros_like(hls_thresh)
    #threshholded[((hls_thresh == 1) & (lab_thresh == 1))& (rgb_thresh==1) & (luv_thresh==1)]=255
    threshholded[((hls_thresh == 1)&(rgb_thresh==1))]=255

    return threshholded


def img_process(img): 
    NUM_SEGS=40
    img=cv2.GaussianBlur(img,(5,5),0)

    pro_img=thresholding(img)
    #img=thresholding(img)
    pro_img = get_middle(pro_img)
    pro_img=dilation(pro_img)
    #segmentCentors, blockCenters = row_segment_centor(pro_img, NUM_SEGS)
    '''
    for i in range(0, NUM_SEGS):
        cv2.circle(img, segmentCentors[i], 5, (255,0,0))
        for j in range(0, len(blockCenters[i])):
            cv2.circle(img, blockCenters[i][j], 5, (0,0,255))
'''
    img=get_middle(img)
    return pro_img, img



def decide_way(img):
    #img=cv2.GaussianBlur(img,(5,5),0)
    blur,croppedImg=img_process(img)

    coor=np.argwhere(blur==255)
    if len(coor) == 0:
        rmean = croppedImg.shape[0]//2
        cmean = croppedImg.shape[1]//2
    else:
        rmean=int(math.floor(np.mean(coor[:,0])))
        cmean=int(math.floor(np.mean(coor[:,1])))
    col=croppedImg.shape[1]//2
    if(cmean<col-30):
        command='Left'
    elif(cmean>col+30):
        command='Right'
    else:
        command='Straight'
    cv2.putText(croppedImg,command, (10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.rectangle(croppedImg,(cmean-20,rmean-20),(cmean+20,rmean+20),(0,255,0),3)
    return command,croppedImg, blur

def capture_and_decide(filename):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    command, img, blur = decide_way(frame)
    cv2.imwrite(filename, img)
    cv2.imwrite("output" + filename, blur)
    print(command)
    return command

#folder='mobot/'
#video='output2.avi'
#image='./sample_pictures/320.jpg'

#start=time.time()
#img=cv2.imread(image)
#command,img,blur=decide_way(img)
#end=time.time()
#print(end-start)
#cv2.imshow('frame',img)
#cv2.imshow('f1',blur)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


'''
cap=cv2.VideoCapture(video)
#img=cv2.imread(os.path.join(folder,filename))
while(cap.isOpened()):
    ret,frame=cap.read()
    command,img,blur=decide_way(frame)
    cv2.imshow('frame',blur)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
