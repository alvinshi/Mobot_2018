from mobot import *

import wiringpi

mobot = Mobot()
mobot.start()
wiringpi.delay(2000)
mobot.go_stop()
