import keras
import time
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import RMSprop, SGD, Adam
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
import acions
from acions import COMPLEX_MOVEMENT

train_dir = os.getcwd() + "/capture_2019-02-181550449590.9676151/"
train_dir2 = os.getcwd() + "/capture_2019-02-181550449813.0011759/"
train_dir3 = os.getcwd() + "/capture_2019-02-181550451211.297548/"
train_dir4 = os.getcwd() + "/capture_2019-02-181550451582.5061865/"
train_dir5 = os.getcwd() + "/capture_2019-02-181550452167.9069276/"
train_dir6 = os.getcwd() + "/capture_2019-02-181550453339.2360086/"
train_dir7 = os.getcwd() + "/capture_2019-02-181550454314.3831165/"
train_dir8 = os.getcwd() + "/capture_2019-02-181550455567.0689166/"
train_dir9 = os.getcwd() + "/capture_2019-02-181550493968.6799326/"
train_dir10 = os.getcwd() + "/capture_2019-02-181550494034.5256565/"
train_dir11 = os.getcwd() + "/capture_2019-02-181550494075.4804735/"
train_dir12 = os.getcwd() + "/capture_2019-02-181550494233.7184389/"
train_dir13 = os.getcwd() + "/capture_2019-02-181550494273.7525651/"
train_dir14 = os.getcwd() + "/capture_2019-02-181550500969.1575031/"
train_dir15 = os.getcwd() + "/capture_2019-02-181550501116.478556/"
train_dir16 = os.getcwd() + "/capture_2019-02-181550501167.8459315/"
train_dir17 = os.getcwd() + "/capture_2019-02-181550501731.0557883/"
train_dir18 = os.getcwd() + "/capture_2019-02-181550502317.1206222/"
train_dir19 = os.getcwd() + "/capture_2019-02-181550504329.6619065/"
train_dir20 = os.getcwd() + "/capture_2019-02-181550504390.5978932/"


csv_train = pd.read_csv(train_dir + "controller_capture.csv")
csv_train2 = pd.read_csv(train_dir2 + "controller_capture.csv")
csv_train3 = pd.read_csv(train_dir3 + "controller_capture.csv")
csv_train4 = pd.read_csv(train_dir4 + "controller_capture.csv")
csv_train5 = pd.read_csv(train_dir5 + "controller_capture.csv")
csv_train6 = pd.read_csv(train_dir6 + "controller_capture.csv")
csv_train7 = pd.read_csv(train_dir7 + "controller_capture.csv")
csv_train8 = pd.read_csv(train_dir8 + "controller_capture.csv")
csv_train9 = pd.read_csv(train_dir9 + "controller_capture.csv")
csv_train10 = pd.read_csv(train_dir10 + "controller_capture.csv")
csv_train11 = pd.read_csv(train_dir11 + "controller_capture.csv")
csv_train12 = pd.read_csv(train_dir12 + "controller_capture.csv")
csv_train13 = pd.read_csv(train_dir13 + "controller_capture.csv")
csv_train14 = pd.read_csv(train_dir14 + "controller_capture.csv")
csv_train15 = pd.read_csv(train_dir15 + "controller_capture.csv")
csv_train16 = pd.read_csv(train_dir16 + "controller_capture.csv")
csv_train17 = pd.read_csv(train_dir17 + "controller_capture.csv")
csv_train18 = pd.read_csv(train_dir18 + "controller_capture.csv")
csv_train19 = pd.read_csv(train_dir19 + "controller_capture.csv")
csv_train20 = pd.read_csv(train_dir20 + "controller_capture.csv")


test_dir = os.getcwd() + "/capture_2019-02-181550450636.0608828/"
test_dir2 = os.getcwd() + "/capture_2019-02-181550504476.5524025/"
csv_test = pd.read_csv(test_dir + "controller_capture.csv")
csv_test2 = pd.read_csv(test_dir2 + "controller_capture.csv")

train_images = [train_dir+i for i in os.listdir(train_dir) if '.png' in i]
train_images += [train_dir2+i for i in os.listdir(train_dir2) if '.png' in i]
train_images += [train_dir3+i for i in os.listdir(train_dir3) if '.png' in i]
train_images += [train_dir4+i for i in os.listdir(train_dir4) if '.png' in i]
train_images += [train_dir5+i for i in os.listdir(train_dir5) if '.png' in i]
train_images += [train_dir6+i for i in os.listdir(train_dir6) if '.png' in i]
train_images += [train_dir7+i for i in os.listdir(train_dir7) if '.png' in i]
train_images += [train_dir8+i for i in os.listdir(train_dir8) if '.png' in i]
train_images += [train_dir9+i for i in os.listdir(train_dir9) if '.png' in i]
train_images += [train_dir10+i for i in os.listdir(train_dir10) if '.png' in i]
train_images += [train_dir11+i for i in os.listdir(train_dir11) if '.png' in i]
train_images += [train_dir12+i for i in os.listdir(train_dir12) if '.png' in i]
train_images += [train_dir13+i for i in os.listdir(train_dir13) if '.png' in i]
train_images += [train_dir14+i for i in os.listdir(train_dir14) if '.png' in i]
train_images += [train_dir15+i for i in os.listdir(train_dir15) if '.png' in i]
train_images += [train_dir16+i for i in os.listdir(train_dir16) if '.png' in i]
train_images += [train_dir17+i for i in os.listdir(train_dir17) if '.png' in i]
train_images += [train_dir18+i for i in os.listdir(train_dir18) if '.png' in i]
train_images += [train_dir19+i for i in os.listdir(train_dir19) if '.png' in i]
train_images += [train_dir20+i for i in os.listdir(train_dir20) if '.png' in i]


test_images = [test_dir+i for i in os.listdir(test_dir) if '.png' in i]
test_images += [test_dir2+i for i in os.listdir(test_dir2) if '.png' in i]
test_list = [csv_test, csv_test2]
csv_test = pd.concat(test_list, axis=0, ignore_index=True)

list_ = [csv_train, csv_train2, csv_train3, csv_train4, csv_train5, csv_train6, csv_train7, csv_train8, csv_train9,
         csv_train10, csv_train11, csv_train12, csv_train13, csv_train14, csv_train15, csv_train16, csv_train17,
         csv_train18, csv_train19, csv_train20]
csv_train = pd.concat(list_, axis=0, ignore_index=True)

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
        fitness.append(reward)
        # print(info)

    env.close()


env = createenvironment(environmentsmario[0], movements[0])

startemulator(env, model, input_shape)
