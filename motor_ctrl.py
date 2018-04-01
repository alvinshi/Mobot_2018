import wiringpi

# Global Constants
OUTPUT = 1
INPUT = 0
PWM_OUTPUT = 2

HIGH = 1
LOW = 0

IN1 = 0
IN2 = 2
ENA = 1
IN3 = 4
IN4 =5
ENB =23

def init():
    #print(wiringpi.piBoardRev())
    wiringpi.wiringPiSetup()
    #print(wiringpi.wpiPinToGpio(1))
    wiringpi.pinMode(IN1, OUTPUT)
    wiringpi.pinMode(IN2, OUTPUT)
    wiringpi.pinMode(ENA, PWM_OUTPUT)
    wiringpi.pinMode(IN3, OUTPUT)
    wiringpi.pinMode(IN4, OUTPUT)
    wiringpi.pinMode(ENB, PWM_OUTPUT)

    #Setup PWM using Pin, Initial Value and Range Parameters
    wiringpi.softPwmCreate(ENA, 0, 100)
    wiringpi.softPwmCreate(ENB, 0, 100)

def go_ahead():
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, HIGH)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, HIGH)

def go_back():
    wiringpi.digitalWrite(IN1, HIGH)
    wiringpi.digitalWrite(IN2, LOW)
    wiringpi.digitalWrite(IN3, HIGH)
    wiringpi.digitalWrite(IN4, LOW)

def go_stop():
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, LOW)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, LOW)

def turn_left():
    wiringpi.digitalWrite(IN1, HIGH)
    wiringpi.digitalWrite(IN2, LOW)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, HIGH)

def turn_right():
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, HIGH)
    wiringpi.digitalWrite(IN3, HIGH)
    wiringpi.digitalWrite(IN4, LOW)

def set_motorspeed(lspeed, rspeed):
    if (lspeed >= 0) and (lspeed < 100):
        wiringpi.softPwmWrite(ENA, lspeed)
    if (rspeed >= 0) and (rspeed < 100):
        wiringpi.softPwmWrite(ENB, rspeed)
        pass

def start():
    set_motorspeed(99, 99)
    go_ahead()

# Test
init()
go_stop()
start()
wiringpi.delay(2000)
go_stop()
#turn_left()
#wiringpi.delay(2000)
#go_stop()
#turn_right()
#wiringpi.delay(2000)
#go_stop()
#set_motorspeed(50, 50)
#go_back()
#wiringpi.delay(2000)
#go_stop()
