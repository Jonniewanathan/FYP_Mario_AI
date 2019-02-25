from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from acions import COMPLEX_MOVEMENT
import acions
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
# train_dir = os.getcwd() + "/capture_2019-02-171550446316.532347/"
# csv_train = pd.read_csv(train_dir + "controller_capture.csv")
# csv_train = csv_train.drop(csv_train.columns[0], axis=1)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)


def convert_to_gray_scale(img):
    return np.dot(img, [0.299, 0.587, 0.114])


def process_image(image, x, y, h, w):
    image = image[y:y + h, x:x + w]
    image = convert_to_gray_scale(image)
    return image


# array = csv_train.get_values()
x_list = []
done = True
action = 0
for step in range(10):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(1)
    env.render()
    print(info['x_pos'])
    x = info['x_pos'] - 22
    y = env.unwrapped._y_position
    print(env.unwrapped._y_position)
    print(env.unwrapped._x_position)
    h = 40 # 52
    w = 60 # 20
    print(env.observation_space.shape)
    x_list.append(info['x_pos'])
    image = env.render('rgb_array')
    image = image[y:y+h, x:x+w]
    # image = cv2.resize(image, dsize=(128, 120), interpolation=cv2.INTER_CUBIC)
    print(image.shape)
    image = convert_to_gray_scale(image)
    print(image.shape)
    # plt.imshow(image, cmap='gray')
    # plt.show()
    plt.imsave(os.getcwd() + "/models/", image, cmap='gray')
    # cv2.imshow('thing', image)
    time.sleep(10)

    # counter = 400 - info['time']
    # action = acions.calculate_action_num(array[counter + 14])
    # print(counter)

env.close()

print(x_list)

