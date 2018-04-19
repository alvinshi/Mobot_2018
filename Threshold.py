import numpy as np
import cv2
import copy
import math

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

def lab_select(img, channel = 'l', thresh=(0, 255)):
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

def hls_select(img, channel = 'l', thresh=(0, 255)):
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
        binary_output[(s_channel < thresh[0]) | (s_channel> thresh[1])] = 1
    return binary_output

def rgb_select(img,thresh=(0,255)):
    ch0=img[:,:,0]
    ch1=img[:,:,1]
    ch2=img[:,:,2]
    binary_output=np.zeros_like(ch0)
    binary_output[(ch0>thresh[0])&(ch0<thresh[1])&(ch1>thresh[0])&(ch1<thresh[1])&(ch2>thresh[0])&(ch2<thresh[1])]=1
    return binary_output

# Dilation to expand white line after thresholding
def dilation(img):
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    return img_dilation

def thresholding(img):
    rgb_thresh = rgb_select(img,(150,255))
    hls_thresh = hls_select(img,channel='l', thresh=(160,240)) #Test Set 2
    #hls_thresh = hls_select(img,channel='l', thresh=(180,240)) #Test Set 1
    #lab_thresh = lab_select(img, channel='l',thresh=(190, 240))
    #luv_thresh = luv_select(img, channel='l',thresh=(180, 240))
    threshholded = np.zeros_like(hls_thresh)
    #threshholded[((hls_thresh == 1) & (lab_thresh == 1))& (rgb_thresh==1) & (luv_thresh==1)]=255
    threshholded[((hls_thresh == 1)&(rgb_thresh==1))]=255
    return dilation(threshholded)
