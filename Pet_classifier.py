import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
# datadir = "./PetImages"
# categories = ["Dog", "Cat"]
# IMG_SIZE = 100
# training_data = []

# def create_training_data():
# 	for animal in categories:
# 		path = os.path.join(datadir, animal)
# 		class_num = categories.index(animal)
		
# 		for img in tqdm(os.listdir(path)):
# 			try:
# 				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
# 				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# 				training_data.append([new_array, class_num])
# 			except Exception as e:
# 				pass

# create_training_data()
# random.shuffle(training_data)
# X = []
# Y = []

# for features, label in training_data:
# 	X.append(features)
# 	Y.append(label)

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# pickle_out = open("X.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("Y.pickle", "wb")
# pickle.dump(Y, pickle_out)
# pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)

X = np.array(X/255.0)
Y = np.array(Y)

dense_layers = [0] #, 1, 2]
layer_sizes = [64] #, 32, 128]
conv_layers = [3] #, 1, 2]

# for dense_layer in dense_layers:
# 	for layer_size in layer_sizes:
# 		for conv_layer in conv_layers:
# 			NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), input_shape = X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(conv_layers[0]-1):
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# for i in range(dense_layer):
# model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, batch_size=32, epochs=3, validation_split=0.1) #, callbacks=[tensorboard])

model.save("Dog_Cat_CNN.model")