
import cv2
import pickle
import face_recognition
import imutils
from imutils.video import VideoStream
from time import sleep
from threading import Thread

data = pickle.loads(open("../face_recognition/encodings.pkl","rb").read())
face_cascade = cv2.CascadeClassifier('../face_recognition/haarcascade_frontalface_default.xml')

def recognize_faces(encodings):
	names=[]
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		name="Unknown"
		if True in matches:
			matchIdx = [i for (i,b) in enumerate(matches) if b]
			counts = {}

			for i in matchIdx:
				name=data["names"][i]
				counts[name] = counts.get(name,0) + 1

			name = max(counts,key=counts.get)
		names.append(name)

#cam = cv2.VideoCapture(0)
cam = VideoStream(src=0).start()

while True:
	img = cam.read()
#	ret, img = cam.read()
	img = cv2.flip(img, 0)
	img = cv2.flip(img, 1)
	img = imutils.resize(img,width=500)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	boxes = [(y,x+w,y+h,x) for (x,y,w,h) in faces]
	encodings = face_recognition.face_encodings(rgb,boxes)
	p1=Thread(target=recognize_faces, args=(encodings,))
	p1.setDaemon(True)
	p1.start()

	for((top,right,bottom,left),name) in zip(boxes,names):
		cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)

		y = top-15 if top-15>15 else top+15
		cv2.putText(img,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

	cv2.imshow('webcam', img)
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cv2.destroyAllWindows()
cam.stop()
