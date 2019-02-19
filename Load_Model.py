import gym_super_mario_bros
import multiprocessing as mp
import keras
import time
import numpy as np
import cv2
import os
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


class Env(object):
    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.done = False

    def calculateaction(self, number, action):
        new_number = float(number[0]) * len(action)
        new_number = round(new_number) - 1

        if new_number < 0:
            new_number = 0

    def run(self):
        classifier = keras.models.load_model(os.getcwd() + "/models/model_732.0")
        action = 0
        old_x_pos = 10000000
        for step in range(5000):
            time.sleep(0.04)
            if self.done:
                state = self.env.reset()
                self.done = False
            state, reward, self.done, info = self.env.step(action)
            self.env.render()
            image = self.env.render('rgb_array')
            image = cv2.resize(image, dsize=(32, 30), interpolation=cv2.INTER_CUBIC)
            image = np.expand_dims(image, axis=0)
            action = np.argmax(classifier.predict(image))
            # action = self.calculateaction(action, SIMPLE_MOVEMENT)
            print(action)
            print(info)
            if step % 120 == 0:
                if info['x_pos'] == old_x_pos:
                    self.done = True
                    old_x_pos = 10000000
                else:
                    old_x_pos = info['x_pos']

        self.env.close()


def runner():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = Env(env)
    env.run()


if __name__ == '__main__':
    procs = []

    for idx in range(1):
        proc = mp.Process(target=runner)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
