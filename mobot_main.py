import time

from mobot import *
import line_following as lf

m = Mobot()
m.start()
seq = 1
while True:
	command = lf.capture_and_decide(str(data.seq) + ".jpg")
	if command == "Straight": data.mobot.go_ahead()
    elif command == "Left": data.mobot.turn_left()
    elif command == "Right": data.mobot.turn_right()
    print("COMMAND: " + command)
    print("Mobot State: " + m.state)
    print("lspeed: " + m.lspeed)
    print("rspeed: " + m.rspeed)
    time.sleep(0.2)