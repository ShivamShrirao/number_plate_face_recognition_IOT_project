#!/usr/bin/env python3
import RPi .GPIO as GPIO
from time import sleep

GREEN_LED=36
RED_LED=29

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(GREEN_LED,GPIO.OUT)
GPIO.setup(RED_LED,GPIO.OUT)

GPIO.output(GREEN_LED,GPIO.HIGH)
GPIO.output(RED_LED,GPIO.HIGH)
sleep(1)
GPIO.output(GREEN_LED,GPIO.LOW)
GPIO.output(RED_LED,GPIO.LOW)
