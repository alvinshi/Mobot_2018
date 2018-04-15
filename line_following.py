import numpy as np
import cv2
import copy
import math
import os

threshold=100

def extract_line(img):
    for row in range(0,len(img)):
        for col in range(0,len(img[row])):
            if(img[row][col]>150):
                img[row][col]=255
            else:
                img[row][col]=0
    return img

def extract_white(new):
    for row in range(0,len(new)):
        for col in range(0,len(new[row])):
            if(img[row][col][2]<200 or img[row][col][1]<200
                    and img[row][col][0]<200):
                new[row][col][2]=0
                new[row][col][1]=0
                new[row][col][0]=0
    return new

def double_raster(img):
    cur_label=2
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
    for r in range(0,len(img)):
        for c in range(0,len(img[r])):
            if(img[r][c]!=0):
                if(eq[img[r][c]]!=0):
                    img[r][c]=eq[img[r][c]]

    return img

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

def thresholding(img):
    #x_thresh = sobel_thresh(img, orient='x', min=30,max=150)
    #mag_thr = mag_thresh(img, sobel_kernel=3, mag_thresh=(40, 150))
    #dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.8, 1.2))
    #rgb_thresh = rgb_select(img,(210,220))
    hls_thresh = hls_select(img, thresh=(40, 60))
    lab_thresh = lab_select(img, thresh=(140, 220))
    luv_thresh = luv_select(img, thresh=(170, 255))
    threshholded = np.zeros_like(hls_thresh)
    threshholded[((hls_thresh == 1) & (lab_thresh == 1)) & (luv_thresh==1)]=255
    cv2.imshow("threshed", threshholded)
    return threshholded

def decide_way(img):
    img=cv2.GaussianBlur(img,(5,5),0)
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


cap=cv2.VideoCapture('./videos/out.h264')
while(True):
    ret,frame=cap.read()
    command,img=decide_way(frame)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
folder='mobot/'
count=0
for filename in os.listdir(folder):
    print filename
    img=cv2.imread(os.path.join(folder,filename))
    img=decide_way(img)
    cv2.imwrite(filename,img)
'''
