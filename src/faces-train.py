import os
import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Points to the base directory
image_dir = os.path.join(BASE_DIR, "images")  #Accesses the images folder


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
            print(label_id)
            #print(label, path)
            #Appending into arrays

            pil_image = Image.open(path).convert("L") #This will convert images into grayscale
            image_array = np.array(pil_image, "uint8")
            print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w] 
                x_train.append(roi) #Appends the region which is to be trained in numpy array format.
                y_labels.append(id_) #Gets the ids of all labels

print(x_train)
print(y_labels)