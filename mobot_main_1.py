import threading
import time
import Queue
import cv2

from mobot import *
import ImageProcess as ip

PAUSE = 0.2
PAST_STATES = 5
LEFT = "left"
RIGHT = "right"

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

m = Mobot(60)

# Variable initialization
seq_n = 1
class Struct(object): pass
data = Struct()
data.img = None
data.stopped = False
data.pastStates = []
data.choices = [RIGHT, LEFT]


v_thread = VideoThread(1, "Thread-1", data)
v_thread.start()
time.sleep(1) # Wait for the camera to adjust its exposure

m.start() # Start the robot
try:
        while not data.stopped:
                img = data.img
                command, frameAtIntersection = ip.get_command(img, data.pastStates, data.choices, seq_n)
                seq_n += 1
                if len(data.pastStates) == PAST_STATES: del data.pastStates[0]
                data.pastStates.append(frameAtIntersection)
                if command == "Straight": m.go_ahead()
                elif command == "Left": m.turn_left()
                elif command == "Right": m.turn_right()
                #print("COMMAND: " + command)
                #print("Mobot State: " + m.state)
                #print("lspeed: " + str(m.lspeed))
                #print("rspeed: " + str(m.rspeed))
except KeyboardInterrupt:
        print("Ctrl + C received")
        data.stopped = True
        m.go_stop()

print("Threads terminated")
