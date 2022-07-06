import cv2 
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={"person_name": 1}
with open("labels.pickle", "rb") as f:   #wb is used as 'write byte'
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0) #Accesses the main camera of the system

while True:
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #Roi-> Region of Interest, only the region we need in whole image
        roi_color = frame[y:y+h, x:x+w]

        #Recognizer, Can also be done by DL using tensorflow or keras but here we will use only opencv
        id_, conf = recognizer.predict(roi_gray) #Conf is confidence of how true it is.
        if conf>=45:  #Its usually from 0-100 but sometimes it crosses 100 and to prevent that we set range to be 45-85
            #print(id_)
            #print(labels[id_])
            #Putting font on the around the roi area
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "1.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) #OpenCV is BGR and not RGB
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()