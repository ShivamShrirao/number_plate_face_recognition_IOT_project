import cv2
import numpy as np
import pytesseract
from threading import Thread

COLOR = (0,255,0)

ssdnet = cv2.dnn.readNetFromTensorflow('trained_model/frozen_inference_graph.pb','trained_model/graph.pbtxt')

# cam = cv2.VideoCapture('vid.mp4')
cam = cv2.VideoCapture(0)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 640,480)

KNOWN_PLATES=["DL3CAM0857","HR26DK8337","MH12DE1433"]

def get_text():
	global plate,splate,text
	plate = cv2.GaussianBlur(plate, (7, 7), 0)
	plate = cv2.erode(plate, (4, 4))
	plate = cv2.dilate(plate, (4, 4))
	splate = cv2.adaptiveThreshold(plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
	text=pytesseract.image_to_string(splate,lang='eng',config="--oem 0 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	if len(text)>0 and len(text)<16:
		print("Plate:",text)
		verify_plate()

def verify_plate():
	global text,VERIFIED
	for pl in KNOWN_PLATES:
		goch=0
		for key in pl:
			if key in text[goch:]:
				goch+=1
		if goch/len(pl)>=0.7:
			VERIFIED=True
			print("Number Plate Verified",pl,goch/len(pl)*100,"%")
VERIFIED=False
text=""
f=0
crop=10
while cam.isOpened():
	ret, img = cam.read()
	f+=1
	if not f%10:
		f=0
		# img = cv2.flip(img, 0)
		# img = cv2.flip(img, 1)
		# img = cv2.resize(img,(640,480))
		rows,cols,channels = img.shape

		ssdnet.setInput(cv2.dnn.blobFromImage(img,size=(400,400),swapRB=True,crop=False))
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
				top=int(detection[4]*rows)-10
				right=int(detection[5]*cols)+10
				bottom=int(detection[6]*rows)+15

				cv2.rectangle(img, (left, top), (right, bottom), COLOR, 2)
				cv2.putText(img, str(score*100)[:5], (left, top),cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)
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
			break
		cv2.imshow("output", img)
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
