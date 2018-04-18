from Tkinter import *

from mobot import *
# import line_following as lf

####################################
# customize these functions
####################################

def init(data):
    data.mobot = Mobot()
    data.cv_command = ""

def keyPressed(event, data, root):
    charText = event.char
    keysymText = event.keysym
    if (keysymText == "space") and ((data.mobot.state == "Stopped") or (data.mobot.state == "Waiting")):
    	data.mobot.start()
    elif keysymText == "space":
        data.mobot.go_stop()
    elif keysymText == "w":
    	data.mobot.go_ahead()
    elif keysymText == "s":
    	data.mobot.go_back()
    elif keysymText == "a":
    	data.mobot.turn_left()
    elif keysymText == "d":
    	data.mobot.turn_right()
    elif keysymText == "k":
        data.mobot.speed_up()
    elif keysymText == "l":
        data.mobot.speed_down()
    elif keysymText == "n":
        data.mobot.toggle_led_left()
    elif keysymText == "m":
        data.mobot.toggle_led_right()
    else:
        root.quit()

def timerFired(data):
    pass
    # data.seq = data.seq + 1
    # command = lf.capture_and_decide(str(data.seq) + ".jpg")
    # data.cv_command = command
    # if command == "Straight": data.mobot.go_ahead()
    # elif command == "Left": data.mobot.turn_left()
    # elif command == "Right": data.mobot.turn_right()

def redrawAll(canvas, data):
    canvas.create_text(data.width/8, data.height/5, text="State: " + data.mobot.state)
    canvas.create_text(data.width/8, data.height/5 * 2, text="Left Wheel Speed: " + str(data.mobot.lspeed))
    canvas.create_text(data.width/8, data.height/5 * 3, text="Right Wheel Speed: " + str(data.mobot.rspeed))
    canvas.create_text(data.width/8, data.height/5 * 4, text="CV Command: " + data.cv_command)
        
####################################
# use the run function as-is
####################################

def run(width=300, height=300, cv = True):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def keyPressedWrapper(event, canvas, data, root):
        keyPressed(event, data, root)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 200
    data.seq = 0
    root = Tk()
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data, root))
    redrawAll(canvas, data)
    data.mobot.start()
    if cv: timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed

run(400, 200, True)
