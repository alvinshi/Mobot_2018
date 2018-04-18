import time

from mobot import *
import line_following as lf

m = Mobot()
#m.start()
seq = 1
while True:
        command = lf.capture_and_decide(str(seq) + ".jpg")
        seq = seq + 1
        #if command == "Straight": m.go_ahead()
        #elif command == "Left": m.turn_left()
        #elif command == "Right": m.turn_right()
        print("COMMAND: " + command)
        print("Mobot State: " + m.state)
        print("lspeed: " + str(m.lspeed))
        print("rspeed: " + str(m.rspeed))
        time.sleep(0.2)
