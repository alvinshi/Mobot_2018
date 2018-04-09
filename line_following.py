import numpy as np
import cv2
import copy

threshold=100

def extract_line(img):
    #new=copy.deepcopy(img)
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
                    up=0;
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
        for c in range(0,len(img[row])):
            if(img[row][col]!=0):
                if(eq[img[row][col]]!=0):
                    img[row][col]=eq[img[row][col]]
    return img

'''
cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
img=cv2.imread('mobot/18.jpg')
#img=extract_white(img)
kernel=np.ones((20,20),np.float32)/400
#hsv=cv2.medianBlur(img,8)
hsv=cv2.filter2D(img,-1,kernel)
#hsv=extract_white(hsv)
gray=cv2.cvtColor(hsv,cv2.COLOR_RGB2GRAY)
gray=extract_line(gray)
'''
edges = cv2.Canny(gray,50,150,apertureSize =5) 
lines=cv2.HoughLines(edges,1,np.pi/180,200)
for r,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)
    '''
#gray=extract_line(gray)
#dr=double_raster(binary)

cv2.imshow('frame',gray)
cv2.waitKey(0);
cv2.destroyAllWindows()

