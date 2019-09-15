#!/usr/bin/env python3
import RPi .GPIO as GPIO
from time import sleep
import sys

MOTOR_PIN1=5
MOTOR_PIN2=7

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(MOTOR_PIN1,GPIO.OUT)
GPIO.setup(MOTOR_PIN2,GPIO.OUT)

if int(sys.argv[1]):
	GPIO.output(MOTOR_PIN1,GPIO.LOW)
	GPIO.output(MOTOR_PIN2,GPIO.HIGH)
else:
	GPIO.output(MOTOR_PIN1,GPIO.HIGH)
	GPIO.output(MOTOR_PIN2,GPIO.LOW)
sleep(1.4)
GPIO.output(MOTOR_PIN1,GPIO.LOW)
GPIO.output(MOTOR_PIN2,GPIO.LOW)
