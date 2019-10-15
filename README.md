# Number Plate Reader and Face recognintion.

Mini IOT project I made for number plate detection, reading and face recognition for authorization using RaspberryPi. Firstly it uses an Ultrasonic sensor to see if there's a car. If there is it starts the camera. I have then trained an SingleShotDetector mobilenetV2 deep learning Object detection network to detect the position of Number Plate. The image is then preprocessed with openCV and uses pytesseract to extract text from image. If the number plate is verified it will do face recognition. For that it uses dlib library which is pretrained on 3 million images to generate encodings of face. Then compares that to the ones trained in database. If verified the garage door will open.

https://www.youtube.com/watch?v=F_sAMYax1J0

## System Diagram

![System Diagram](/diagram.png)

## Click Below for Demo

[![System Video](/project.jpeg)](https://www.youtube.com/watch?v=F_sAMYax1J0 "Click to watch video.")

![Plate Read](/num_plate_read.png)

![Face Recognition](/face_rec.png)

![Plate Read](/plate_sys.jpeg)

![Face Rec](/face_sys.jpeg)

![Plate detect](/plate_detect.jpeg)

Face Recognition backend link: https://github.com/ShivamShrirao/face_recognition