#!/usr/bin/env python3
import RPi .GPIO as GPIO
import time

TRIGGER=40
ECHO=38

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(TRIGGER,GPIO.OUT)

while True:
	GPIO.output(TRIGGER,True)
	time.sleep(0.00001)
	GPIO.output(TRIGGER,False)

	while GPIO.input(ECHO)==0:
		start = time.time()
	while GPIO.input(ECHO)==1:
		stop = time.time()

	time_elap = stop-start
	distance = time_elap*17150
	print("Calculated:",distance)
	time.sleep(1)
