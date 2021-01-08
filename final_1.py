from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

# Specifying Categories
categories = ["camera","lamp","helicopter","laptop","stapler"]
nb_classes = len(categories)

# Specify image size
image_w = 64 
image_h = 64

# Load the Data
X_train, X_test, y_train, y_test = np.load("5object.npy", allow_pickle=True)

# normalize the data
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# Build the Model
in_shape = X_train.shape[1:]
model = Sequential()
model.add(Conv2D(32, 3, input_shape=in_shape)) 
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes)) 
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# train a model 
model.fit(X_train, y_train, batch_size=24, epochs=40)

# evaluate a model
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])

# Analyze image1 photos (Stapler)
from keras.preprocessing import image
img = image.load_img("image1.png",target_size=(64,64,3))
img = image.img_to_array(img)
img = img/256
img = img.reshape(1,64,64,3)
predict = model.predict(img)
print (predict)
