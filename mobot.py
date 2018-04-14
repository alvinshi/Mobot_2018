import wiringpi

class Mobot:
	INPUT = 0
	OUTPUT = 1
	PWM_OUTPUT = 2

	HIGH = 1
	LOW = 0

	def __init__(self, CRUISE_SPEED = 99):
		self.IN1 = 0
		self.IN2 = 2
		self.ENA = 1
		self.IN3 = 4
		self.IN4 = 5
		self.ENB = 23

		self.LEDL = 28
		self.LEDR = 29

		self.MOBOT_AXIS = 24 # In centimeter
		self.CRUISE_SPEED = CRUISE_SPEED

		self.init_gpio()
                self.lled = 0
                self.rled = 0

	def init_gpio(self):
		wiringpi.wiringPiSetup()
		wiringpi.pinMode(self.IN1, self.OUTPUT)
		wiringpi.pinMode(self.IN2, self.OUTPUT)
		wiringpi.pinMode(self.ENA, self.PWM_OUTPUT)
		wiringpi.pinMode(self.IN3, self.OUTPUT)
		wiringpi.pinMode(self.IN4, self.OUTPUT)
		wiringpi.pinMode(self.ENB, self.PWM_OUTPUT)

		#Setup PWM using Pin, Initial Value and Range Parameters
		wiringpi.softPwmCreate(self.ENA, 0, 100)
		wiringpi.softPwmCreate(self.ENB, 0, 100)

                #LED
                wiringpi.pinMode(self.LEDL, self.OUTPUT)
                wiringpi.pinMode(self.LEDR, self.OUTPUT)
                
		self.lspeed = 0
		self.rspeed = 0
		self.state = "Waiting"

        def toggle_led_left(self):
                self.lled = not self.lled
                wiringpi.digitalWrite(self.LEDL, self.lled)

        def toggle_led_right(self):
                self.rled = not self.rled
                wiringpi.digitalWrite(self.LEDR, self.rled)

	def go_ahead(self):
		self.go_stop()
		speed = max(self.lspeed, self.rspeed)
		self.set_motorspeed(speed, speed)
		wiringpi.digitalWrite(self.IN1, self.LOW)
		wiringpi.digitalWrite(self.IN2, self.HIGH)
		wiringpi.digitalWrite(self.IN3, self.LOW)
		wiringpi.digitalWrite(self.IN4, self.HIGH)
		self.state = "Forward"

	def go_back(self):
		self.go_stop()
		speed = max(self.lspeed, self.rspeed)
		self.set_motorspeed(speed, speed)
		wiringpi.digitalWrite(self.IN1, self.HIGH)
		wiringpi.digitalWrite(self.IN2, self.LOW)
		wiringpi.digitalWrite(self.IN3, self.HIGH)
		wiringpi.digitalWrite(self.IN4, self.LOW)
		self.state = "Backward"

	def go_stop(self):
		wiringpi.digitalWrite(self.IN1, self.LOW)
		wiringpi.digitalWrite(self.IN2, self.LOW)
		wiringpi.digitalWrite(self.IN3, self.LOW)
		wiringpi.digitalWrite(self.IN4, self.LOW)
		self.state = "Stopped"

	def rotate_left(self):
		self.go_stop()
		speed = max(self.lspeed, self.rspeed)
		self.set_motorspeed(speed, speed)
		wiringpi.digitalWrite(self.IN1, self.HIGH)
		wiringpi.digitalWrite(self.IN2, self.LOW)
		wiringpi.digitalWrite(self.IN3, self.LOW)
		wiringpi.digitalWrite(self.IN4, self.HIGH)
		self.state = "Rotating Left"

	def rotate_right(self):
		self.go_stop()
		speed = max(self.lspeed, self.rspeed)
		self.set_motorspeed(speed, speed)
		wiringpi.digitalWrite(self.IN1, self.LOW)
		wiringpi.digitalWrite(self.IN2, self.HIGH)
		wiringpi.digitalWrite(self.IN3, self.HIGH)
		wiringpi.digitalWrite(self.IN4, self.LOW)
		self.state = "Rotating Right"

	def turning_ratio(self, radius):
		ratio = (float(radius + self.MOBOT_AXIS)) / radius
                return ratio

	def turn_left(self, radius = 20):
		self.go_stop()
		ratio = self.turning_ratio(radius)
		lspeed = int(self.rspeed / ratio)
		self.set_motorspeed(lspeed, self.rspeed)
		wiringpi.digitalWrite(self.IN1, self.LOW)
		wiringpi.digitalWrite(self.IN2, self.HIGH)
		wiringpi.digitalWrite(self.IN3, self.LOW)
		wiringpi.digitalWrite(self.IN4, self.HIGH)
		self.state = "Turning Left"

	def turn_right(self, radius = 20):
		self.go_stop()
		ratio = self.turning_ratio(radius)
		rspeed = int(self.lspeed / ratio)
		self.set_motorspeed(self.lspeed, rspeed)
		wiringpi.digitalWrite(self.IN1, self.LOW)
		wiringpi.digitalWrite(self.IN2, self.HIGH)
		wiringpi.digitalWrite(self.IN3, self.LOW)
		wiringpi.digitalWrite(self.IN4, self.HIGH)
		self.state = "Turning Right"

	def set_motorspeed(self, lspeed, rspeed):
		if lspeed < 0: lspeed = 0
		if rspeed < 0: rspeed = 0
		if lspeed >= 100: lspeed = 99
		if rspeed >= 100: rspeed = 99
		wiringpi.softPwmWrite(self.ENA, lspeed)
		self.lspeed = lspeed
		wiringpi.softPwmWrite(self.ENB, rspeed)
		self.rspeed = rspeed

	def start(self):
		self.set_motorspeed(self.CRUISE_SPEED, self.CRUISE_SPEED)
		self.go_ahead()

	def speed_up(self, delta = 20):
		self.set_motorspeed(self.lspeed + delta, self.rspeed + delta)

	def speed_down(self, delta = 20):
		self.set_motorspeed(self.lspeed - delta, self.rspeed - delta)
