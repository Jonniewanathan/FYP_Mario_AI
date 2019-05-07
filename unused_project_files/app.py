from multiprocessing import Pool
import random
import time
import numpy as np
import scipy.misc as smp
import cv2
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

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


def converttograyscale(display):
    return np.dot(display, [0.299, 0.587, 0.114])


def convertimagesize(display, height, width):
    resized_image = cv2.resize(display, (height, width))
    return resized_image
    # return np.resize(display, (height, width))


def displayarray(img):
    img = smp.toimage(img)
    img.show()


def createenvironment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


def startemulator(env):
    done = True
    for step in range(5000):
        time.sleep(0.06)
        if done or step == 0:
            env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        print(info)
        env.render()
    env.close()


environments = []

if __name__ == '__main__':

    for num in range(1):
        environments.append(createenvironment(random.choice(environmentsmario), random.choice(movements)))

    pool = Pool()
    pool.map(startemulator, environments)
    # pool.close()
    # pool.join()




