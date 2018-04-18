import numpy as np
import cv2

cap=cv2.VideoCapture('output2.avi')
index = 0
while(cap.isOpened()):
    ret, frame=cap.read()
    cv2.imshow('frame',frame)
    # if index % 20 == 0:
    #     imageName = './sample_pictures/%d.jpg'%index
    #     cv2.imwrite(imageName, frame)
    index = index + 1
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
