from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np

# choose 5 category to learn
caltech_dir = "image_category"
categories = ["camera","lamp","helicopter","laptop","stapler"]
nb_classes = len(categories)

# decide the image size
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3

# installing the image
X = []
Y = []
for idx, cat in enumerate(categories):

    # Set the label
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # Images
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f) # --- (※6)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

# Divide the learn data and test data
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("5object.npy", xy)

print("ok,", len(Y))




