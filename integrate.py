#!/usr/bin/env python3
import cv2
import pickle
import face_recognition
import imutils
import numpy as np
import pytesseract
import subprocess
import time
import RPi .GPIO as GPIO
from imutils.video import VideoStream
from time import sleep
from threading import Thread
from multiprocessing import Process, Pool

MOTOR_PIN1=5
MOTOR_PIN2=7

TRIGGER=40
ECHO=38
GREEN_LED=36
RED_LED=29

window_res=540,380

face_data = pickle.loads(open("../face_recognition/encodings.pkl","rb").read())
face_cascade = cv2.CascadeClassifier('../face_recognition/haarcascade_frontalface_default.xml')

GREEN = (0,255,0)
RED = (0,0,255)
ssdnet = cv2.dnn.readNetFromTensorflow('trained_model/frozen_inference_graph.pb','trained_model/graph.pbtxt')

KNOWN_PLATES=["HR26DK8337"]

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(TRIGGER,GPIO.OUT)
GPIO.setup(GREEN_LED,GPIO.OUT)
GPIO.setup(RED_LED,GPIO.OUT)

def get_text():
	global plate,splate,text
	plate = cv2.GaussianBlur(plate, (7, 7), 0)
	plate = cv2.erode(plate, (4, 4))
	plate = cv2.dilate(plate, (4, 4))
	splate = cv2.adaptiveThreshold(plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
	text=pytesseract.image_to_string(splate,lang='eng',config="--oem 0 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	if len(text)>0 and len(text)<16:
		print(text)
		verify_plate()

def verify_plate():
	global text,VERIFIED,COLOR
	for pl in KNOWN_PLATES:
		goch=0
		for key in pl:
			if key in text[goch:]:
				goch+=1
		if goch/len(pl)>=0.8:
			VERIFIED=True
			COLOR=GREEN
			print("[*] Number Plate Verified",pl,"\t",goch/len(pl)*100,"%")

def determine_faces(encoding):
	global face_data,COLOR
	matches = face_recognition.compare_faces(face_data["encodings"],encoding)
	name="Unknown"
	if True in matches:
		matchIdx = [i for (i,b) in enumerate(matches) if b]
		counts = {}
		for i in matchIdx:
			name=face_data["names"][i]
			counts[name] = counts.get(name,0) + 1

		name = max(counts,key=counts.get)
		COLOR=GREEN
	return name

def verify_faces(names):
	global VERIFIED_FACE,COLOR,det_names
	for name in names:
		if name!="Unknown":
			det_names[name]+=1
			if det_names[name]>=3:
				VERIFIED_FACE=True
				COLOR=GREEN
				print("[*] Authorization complete with",name,'.')


def unload_buffer():
	global cam
	for i in range(10):
		cam.read()

#cam = cv2.VideoCapture(0)
cam = VideoStream(src=0).start()

VERIFIED=False
VERIFIED_FACE=False
COLOR=RED
det_names=dict.fromkeys(set(face_data["names"]))
def recog_faces():
	global VERIFIED_FACE,COLOR,det_names
	for key in det_names.keys():
		det_names[key]=0
	VERIFIED_FACE=False
	COLOR=RED
	pol=Pool()
	f=0
	while True:
		img = cam.read()
		f+=1
		if not f%10:
			f=0
			t1=Thread(target=unload_buffer)
			t1.setDaemon(True)
			t1.start()
			img = cam.read()
		#	ret, img = cam.read()
			# img = cv2.flip(img, 0)
			# img = cv2.flip(img, 1)
			img = imutils.resize(img,width=400)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			if len(faces)>0:
				boxes = [(y,x+w,y+h,x) for (x,y,w,h) in faces]
				encodings = face_recognition.face_encodings(rgb,boxes)
				names=pol.starmap(determine_faces, [(encoding,) for encoding in encodings])		#Things I do for 'performance', #clusterfk
				t1=Thread(target=verify_faces,args=(names,))
				t1.setDaemon(True)
				t1.start()
				print(names)
			else:
				names=[]
				boxes=[]
			for((top,right,bottom,left),name) in zip(boxes,names):
				cv2.rectangle(img,(left,top),(right,bottom),COLOR,2)
				y = top-15 if top-15>15 else top+15
				cv2.putText(img,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,COLOR,2)
			try:
				cv2.imshow('face_rec', img)
			except:
				pass
			if VERIFIED_FACE:
				print("[!] exitting face_rec")
				break
		key = cv2.waitKey(1) & 0xff
		if key == ord('q'):
			break

text=""
def detect_plates():
	global plate,VERIFIED,COLOR
	VERIFIED=False
	COLOR=GREEN
	f=0
	crop=10
	while True:
		img = cam.read()
		f+=1
		if not f%10:
			f=0
			img = cv2.resize(img, (300, 300))
			rows,cols,channels = img.shape

			ssdnet.setInput(cv2.dnn.blobFromImage(img,size=(300,300),swapRB=True,crop=False))
			netout = ssdnet.forward()

			scores=[]
			for detection in netout[0,0]:
				scores.append(float(detection[2]))

			if len(scores)>2:
				first=np.argmax(scores)
				scores.pop(first)
				second=np.argmax(scores)
				idtxs=[first,second]
			else:
				idtxs = range(len(scores))

			for idx in idtxs:
				detection=netout[0,0][idx]
				score = float(detection[2])
				if score >0.3:
					left=int(detection[3]*cols)
					top=int(detection[4]*rows)-15
					right=int(detection[5]*cols)+10
					bottom=int(detection[6]*rows)+15
					cv2.rectangle(img, (left, top), (right, bottom), COLOR, 2)
					# cv2.putText(img, str(score*100)[:5], (left, top),cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)
					plate = img[top+crop:bottom-crop,left+crop:right-crop]
					try:
						plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
						#p1=Process(target=get_text, args=())
						p1=Thread(target=get_text, args=())
						p1.setDaemon(True)
						p1.start()
						cv2.imshow("plate",splate)
					except:
						pass
			if VERIFIED:
				print("[!] exitting plate_out")
				break
			cv2.imshow("plate_output", img)
		key = cv2.waitKey(1) & 0xff
		if key == ord('q'):
			break

subprocess.call(['/home/pi/number_plate_detection/led.py'])
print("[*] Starting ultrasonic.")
while True:
	time.sleep(1)
	GPIO.output(TRIGGER,True)
	time.sleep(0.00001)
	GPIO.output(TRIGGER,False)

	while GPIO.input(ECHO)==0:
		start = time.time()
	while GPIO.input(ECHO)==1:
		stop = time.time()

	time_elap = stop-start
	distance = time_elap*17150
	print("Distance:",distance)
	if distance<25:
		cv2.namedWindow("plate_output", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("plate_output", *window_res)
		GPIO.output(RED_LED,GPIO.HIGH)
		detect_plates()
		cv2.destroyAllWindows()
		if VERIFIED:
			sleep(1)
			VERIFIED_FACE=False
			cv2.namedWindow("face_rec", cv2.WINDOW_NORMAL)
			cv2.resizeWindow("face_rec", *window_res)
			recog_faces()
			if VERIFIED_FACE:
				GPIO.output(RED_LED,GPIO.LOW)
				GPIO.output(GREEN_LED,GPIO.HIGH)
				cv2.destroyAllWindows()
				subprocess.call(['/home/pi/number_plate_detection/dc_motor.py','1'])
				sleep(4)
				subprocess.call(['/home/pi/number_plate_detection/dc_motor.py','0'])
				break
GPIO.output(GREEN_LED,GPIO.LOW)
sleep(2)
GPIO.output(RED_LED,GPIO.LOW)

cam.stop()
