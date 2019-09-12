import cv2
import numpy as np

ssdnet = cv2.dnn.readNetFromTensorflow('trained_model/frozen_inference_graph.pb','trained_model/graph.pbtxt')

cam = cv2.VideoCapture(0)

while True:
	ret, img = cam.read()
	img = cv2.flip(img, 1)
	rows,cols,channels = img.shape

	ssdnet.setInput(cv2.dnn.blobFromImage(img,size=(400,400),swapRB=True,crop=False))
	netout = ssdnet.forward()

	for detection in netout[0,0]:
		score = float(detection[2])
		if score >0.2:
			left=detection[3]*cols
			top=detection[4]*rows
			right=detection[5]*cols
			bottom=detection[6]*rows

			cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
			cv2.putText(img, label, (left, top),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	
	cv2.imshow('webcam', img)
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()