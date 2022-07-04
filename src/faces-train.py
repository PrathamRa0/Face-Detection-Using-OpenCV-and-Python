import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Points to the base directory
image_dir = os.path.join(BASE_DIR, "images")  #Accesses the images folder

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            #Appending into arrays

            pil_image = Image.open(path).convert("L") #This will convert images into grayscale
            image_array = np.array(pil_image, "uint8")
            print(image_array)