from __future__ import division

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

MOBOT_AXIS = 24 # In centimeter

CRUISE_SPEED = 99

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
    go_stop()
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, HIGH)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, HIGH)

def go_back():
    go_stop()
    wiringpi.digitalWrite(IN1, HIGH)
    wiringpi.digitalWrite(IN2, LOW)
    wiringpi.digitalWrite(IN3, HIGH)
    wiringpi.digitalWrite(IN4, LOW)

def go_stop():
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, LOW)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, LOW)

def rotate_left():
    go_stop()
    wiringpi.digitalWrite(IN1, HIGH)
    wiringpi.digitalWrite(IN2, LOW)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, HIGH)

def rotate_right():
    go_stop()
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, HIGH)
    wiringpi.digitalWrite(IN3, HIGH)
    wiringpi.digitalWrite(IN4, LOW)

def turning_ratio(radius):
    ratio = (radius + MOBOT_AXIS) / radius

def turn_left(radius):
    go_stop()
    ratio = turning_ratio(radius)
    lspeed = int(ratio * CRUISE_SPEED)
    wiringpi.softPwmWrite(ENA, lspeed)
    wiringpi.softPwmWrite(ENB, CRUISE_SPEED)
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, HIGH)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, HIGH)

def turn_right(radius):
    go_stop()
    ratio = turning_ratio(radius)
    rspeed = int(ratio * CRUISE_SPEED)
    wiringpi.softPwmWrite(ENA, CRUISE_SPEED)
    wiringpi.softPwmWrite(ENB, rspeed)
    wiringpi.digitalWrite(IN1, LOW)
    wiringpi.digitalWrite(IN2, HIGH)
    wiringpi.digitalWrite(IN3, LOW)
    wiringpi.digitalWrite(IN4, HIGH)

def set_motorspeed(lspeed, rspeed):
    if (lspeed >= 0) and (lspeed < 100):
        wiringpi.softPwmWrite(ENA, lspeed)
    if (rspeed >= 0) and (rspeed < 100):
        wiringpi.softPwmWrite(ENB, rspeed)
    else:
        raise ValueError("Speed Range is 0 ~ 100")

def start():
    set_motorspeed(CRUISE_SPEED, CRUISE_SPEED)
    go_ahead()

# Test
init()
start()
wiringpi.delay(2000)
go_stop()
