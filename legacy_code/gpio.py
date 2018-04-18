import RPi.GPIO as GPIO

# Set up the GPIO board
GPIO.setmode(GPIO.BOARD)
mode = GPIO.getmode()
print(mode)

# Get the information of the Raspberry Pi
print(GPIO.RPI_INFO)

# Suppress the warning
GPIO.setwarnings(False)

