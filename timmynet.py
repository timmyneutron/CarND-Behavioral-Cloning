from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, \
						 Lambda, Dropout, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2

import numpy as np
from scipy.misc import imresize
import cv2
from preprocess import preprocess, get_preprocess_dict
from sklearn.model_selection import train_test_split
from load import load_data
import pickle

### load data

X_train, y_train = load_data("driving_log.csv")

### preprocess data

preprocess_dict = get_preprocess_dict()
X_train = preprocess(X_train, preprocess_dict)
pickle.dump(preprocess_dict, open("preprocess_dict.p", "wb"))

### split data

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

### build model

model = Sequential()

samples, height, width, channels = X_train.shape

model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(height, width, channels)))

print(model.output_shape)

model.add(Convolution2D(16, 6, 6, border_mode="same", activation="relu"))

model.add(AveragePooling2D(pool_size=(3, 3)))

print(model.output_shape)

model.add(Convolution2D(32, 6, 6, border_mode="same", activation="relu"))

model.add(AveragePooling2D(pool_size=(2, 2)))

print(model.output_shape)

model.add(Convolution2D(64, 6, 6, border_mode="same", activation="relu"))

model.add(AveragePooling2D(pool_size=(2, 2)))

print(model.output_shape)

# model.add(Convolution2D(96, 6, 6, border_mode="same", activation="relu"))

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

print(model.output_shape[1])

model.add(Dropout(.5))

# model.add(Dense(2048, activation="relu"))

# model.add(Dropout(.5))

model.add(Dense(1024, activation="relu"))

model.add(Dropout(.5))

model.add(Dense(1, activation="tanh"))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss="mse")

### train model

nb_train_samples = len(X_train)
nb_validation_samples = len(X_validation)

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=32)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=32)

model.fit_generator(train_generator, samples_per_epoch=nb_train_samples,
					nb_epoch=5, validation_data=validation_generator,
					nb_val_samples=nb_validation_samples)

# mse = model.evaluate_generator(test_generator, val_samples=nb_test_samples)

# print("MSE: {:.4f}".format(mse))

# ### save model

json_string = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(json_string)

model.save_weights('model.h5')
