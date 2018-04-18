import numpy as np
import cv2

cap=cv2.VideoCapture('output2.avi')
while(cap.isOpened()):
    ret, frame=cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
