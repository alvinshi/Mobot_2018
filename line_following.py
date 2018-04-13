import os
import time
import ImageProcessor
import VideoStream
import SignalHandler
import threading
import logging
import signal

# Thread for image processing
class imageProcessorThread(threading.Thread):
    def __init__(self, name, VideoStream, ImageProcessor, stopper):
        threading.Thread.__init__(self)
        self.name = name
        self.stream = VideoStream
        self.processor = ImageProcessor
        self.stopper = stopper
    def run(self):
        while not self.stopper.is_set():
            img = self.stream.getCurrentFrame()
            command, img = self.processor.decide_way(img)
            print(command)

# Thread for video streaming
class videoStreamThread(threading.Thread):
    def __init__(self, name, VideoStream):
        threading.Thread.__init__(self)
        self.name = name
        self.stream = VideoStream
    def run(self):
        self.stream.startVideoStream()
        

if __name__ == "__main__":
    # Create stop signal thread Event
    stopper = threading.Event()

    # Create Instances
    ImageProcessor = ImageProcessor.ImageProcessor()
    VideoStream = VideoStream.VideoStream(stopper, bufferSize=10, videoInput="./videos/out.h264")

    # Create threads
    IPThread = imageProcessorThread("imageProcessor", VideoStream, ImageProcessor, stopper)
    VSThread = videoStreamThread("videoStream", VideoStream)
    threads = [IPThread, VSThread]  # List of all threads, used for the stopping object

    # Create stop signal handler instance
    stopHandler = SignalHandler.SignalHandler(stopper, threads)
    signal.signal(signal.SIGINT, stopHandler)

    # Wait for 2 seconds to start image processing after starting video streaming
    VSThread.start()
    time.sleep(2)
    IPThread.start()
    VSThread.join()
    IPThread.join()





'''
folder='mobot/'
count=0
for filename in os.listdir(folder):
    print filename
    img=cv2.imread(os.path.join(folder,filename))
    img=decide_way(img)
    cv2.imwrite(filename,img)
'''
