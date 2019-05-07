import keras
import time
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import RMSprop, SGD, Adam
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import sys

import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
import acions
from acions import COMPLEX_MOVEMENT
train = os.getcwd() + "\\train\\"
test = os.getcwd() + "\\test\\"
train_directories = [train+i for i in os.listdir(train) if 'capture' in i]
test_directories = [test+i for i in os.listdir(test) if 'capture' in i]
print(train_directories)

csv_train = []
train_images = []

csv_test = []
test_images = []

for _train in train_directories:
    csv_train.append(pd.read_csv(_train + "\\controller_capture.csv"))
    train_images += [_train + "\\" + i for i in os.listdir(_train) if '.png' in i]
csv_train = pd.concat(csv_train, axis=0, ignore_index=True)

for _test in test_directories:
    csv_test.append(pd.read_csv(_test + "\\controller_capture.csv"))
    test_images += [_test + "\\" + i for i in os.listdir(_test) if '.png' in i]
csv_test = pd.concat(csv_test, axis=0, ignore_index=True)

print(train_images)

ROWS = 128
COLS = 120
CHANNELS = 3


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    img2 = cv2.resize(img2, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img2 = np.reshape(img2, (ROWS, COLS, CHANNELS))
    return img2


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))

    return data


train = prep_data(train_images)
test = prep_data(test_images)
# input_shape = read_image(train_images[0]).shape

input_shape = (ROWS, COLS, CHANNELS)

csv_train = csv_train.drop(csv_train.columns[0], axis=1)
csv_test = csv_test.drop(csv_test.columns[0], axis=1)

optimizer = Adam(lr=0.001)
objective = 'binary_crossentropy'

model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

model.fit(train, csv_train, epochs=80, verbose=1)

print(model.evaluate(test, csv_test, verbose=1))


environmentsmario = ["SuperMarioBros-v0",
                     "SuperMarioBros-v1",
                     "SuperMarioBros-v2",
                     "SuperMarioBros-v3",
                     "SuperMarioBrosNoFrameskip-v0",
                     "SuperMarioBrosNoFrameskip-v1",
                     "SuperMarioBrosNoFrameskip-v2",
                     "SuperMarioBrosNoFrameskip-v3",
                     "SuperMarioBros2-v0",
                     "SuperMarioBros2-v1",
                     "SuperMarioBros2NoFrameskip-v0",
                     "SuperMarioBros2NoFrameskip-v1"]

movements = [COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY]

image_shape = (ROWS, COLS, CHANNELS)


def createenvironment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


def startemulator(env, model, image_shape):
    fitness = []
    done = True
    env.reset()
    old_x_pos = sys.maxsize
    for step in range(5000):
        time.sleep(0.05)
        env.render()
        image = env.render('rgb_array')
        image = cv2.resize(image, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, axis=0)
        if done or step == 0:
            print(np.sum(fitness))
            fitness = []
            env.reset()
        action = model.predict(image)
        action = acions.calculate_action_list(action[0])
        action = acions.calculate_action_num(action)
        state, reward, done, info = env.step(action)
        if step % 60 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
            else:
                old_x_pos = info['x_pos']
        fitness.append(reward)
        # print(info)

    env.close()


env = createenvironment(environmentsmario[0], movements[0])

startemulator(env, model, input_shape)
