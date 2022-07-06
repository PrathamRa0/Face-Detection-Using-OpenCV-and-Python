import os
import cv2
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Points to the base directory
image_dir = os.path.join(BASE_DIR, "images")  #Accesses the images folder

recognizer = cv2.face.LBPHFaceRecognizer_create()

curr_id = 0  #To get to know the id of each label
label_id = {} #To get all the labels and store it.
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            
            if label in label_id:
                pass
            else:
                label_id[label] = curr_id
                curr_id += 1
            
            id_ = label_id[label]
            #print(label_id)
            #print(label, path)
            #Appending into arrays

            pil_image = Image.open(path).convert("L") #This will convert images into grayscale
            #Resizing images for training
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(final_image, "uint8")
            #print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w] 
                x_train.append(roi) #Appends the region which is to be trained in numpy array format.
                y_labels.append(id_) #Gets the ids of all labels

#print(x_train)
#print(y_labels)

#Pickle module used for serialization. It saves the code in computer readable format.
#Here we will use pickle to save label IDs.
#Dump and load are the 2 main functions provided by this module

with open("labels.pickle", "wb") as f:   #wb is used as 'write byte'
    pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_labels)) #Training x_train with y_labels. We convert y_label to numpy array before that
recognizer.save("trainner.yml")