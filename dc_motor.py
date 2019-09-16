#!/usr/bin/env python3
import RPi .GPIO as GPIO
from time import sleep
import sys

MOTOR_PIN1=5
MOTOR_PIN2=7
MOT_VCC=37


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(MOTOR_PIN1,GPIO.OUT)
GPIO.setup(MOTOR_PIN2,GPIO.OUT)

GPIO.setup(MOT_VCC,GPIO.OUT)
GPIO.output(MOT_VCC,GPIO.HIGH)

if int(sys.argv[1]):
	GPIO.output(MOTOR_PIN1,GPIO.LOW)
	GPIO.output(MOTOR_PIN2,GPIO.HIGH)
else:
	GPIO.output(MOTOR_PIN1,GPIO.HIGH)
	GPIO.output(MOTOR_PIN2,GPIO.LOW)
sleep(2.2)
GPIO.output(MOTOR_PIN1,GPIO.LOW)
GPIO.output(MOTOR_PIN2,GPIO.LOW)
