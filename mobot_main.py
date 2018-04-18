import threading
import time
import Queue
import cv2

from mobot import *
import line_following as lf

PAUSE = 0.2

class VideoThread(threading.Thread):
        def __init__(self, threadID, name, data):
                threading.Thread.__init__(self)
                self.threadID = threadID
                self.name = name
                self.data = data
                self.cap = cv2.VideoCapture(0)
        def run(self):
                while not self.data.stopped:
                        ret, frame = self.cap.read()
                        data.img = frame

m = Mobot()
seq = 1
class Struct(object): pass
data = Struct()
data.img = None
data.stopped = False
v_thread = VideoThread(1, "Thread-1", data)
v_thread.start()
time.sleep(1) # Wait for the camera to adjust its exposure
m.start()
while not data.stopped:
        try:
                img = data.img
                command, cropped_img, blurred_img = lf.decide_way(img)
                cv2.imwrite(str(seq) + "input.jpg", cropped_img)
                cv2.imwrite(str(seq) + "out.jpg", blurred_img)
                seq = seq + 1
                if command == "Straight": m.go_ahead()
                elif command == "Left": m.turn_left()
                elif command == "Right": m.turn_right()
                print("COMMAND: " + command)
                print("Mobot State: " + m.state)
                print("lspeed: " + str(m.lspeed))
                print("rspeed: " + str(m.rspeed))
        except KeyboardInterrupt:
                print("Ctrl + C received")
                data.stopped = True
                m.go_stop()

print("Threads terminated")
