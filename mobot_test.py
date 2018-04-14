from Tkinter import *

import mobot

####################################
# customize these functions
####################################

def init(data):
	data.mobot = Mobot()
    pass

def keyPressed(event, data):
	charText = event.char
    keysymText = event.keysym
    if keysymText == "space":
    	data.mobot.start()
    elif keysymText == "w":
    	data.mobot.go_ahead()
    elif keysymText == "s":
    	data.mobot.go_back()
    elif keysymText == "a":
    	data.mobot.turn_left()
    elif keysymText == "d":
    	data.mobot.turn_right()

def redrawAll(canvas, data):
	canvas.create_text(data.width/10, data.height/4, text="State: " + data.mobot.state)
	canvas.create_text(data.width/10, data.height/4 * 2, text="Left Wheel Speed: " + data.mobot.lspeed)
	canvas.create_text(data.width/10, data.height/4 * 3, text="Right Wheel Speed: " + data.mobot.rspeed)

####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    root = Tk()
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    redrawAll(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed

run(400, 200)