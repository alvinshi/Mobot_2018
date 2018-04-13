import numpy as np
import cv2
import copy
import math
import os
import threading

class VideoStream:
    def __init__(self, stopper, bufferSize = 10, videoInput = 0):
        self.capture = cv2.VideoCapture(videoInput)
        self.frameBuffer = [None] * bufferSize
        self.bufferSize = bufferSize
        self.bufferIndex = 0
        self.stopper = stopper  # The stopper is for the CtlC stop signal
        self.lock = threading.Lock()    # Lock that prevents simultaneous read and write

    # Put current frame from video stream to buffer
    def __putFrameToBuffer(self, frame):
        pos = self.bufferIndex % self.bufferSize # Cyclic buffer
        self.lock.acquire()
        self.frameBuffer[pos] = frame
        self.bufferIndex += 1
        self.lock.release()

    # Start the video stream and quickly put frame to buffer
    def startVideoStream(self):
        while not self.stopper.is_set():
            ret, frame = self.capture.read()
            if ret:
                self.__putFrameToBuffer(frame)
            else:
                pass
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Get the latest frame from buffer
    def getCurrentFrame(self):
        self.lock.acquire()
        curFrame = self.frameBuffer[self.bufferIndex % self.bufferSize]
        self.lock.release()
        return curFrame
