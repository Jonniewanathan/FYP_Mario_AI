import time
from Image_Classifier import ImageClassifier
from ann_visualizer.visualize import ann_viz;
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


def displayarray(img):
    img = smp.toimage(img)
    img.show()


def createenvironment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


def calculateaction(number, movement):
    newnumber = float(number[0]) * len(movement)
    print(newnumber)
    newnumber = round(newnumber) -1

    if newnumber < 0:
        newnumber = 0

    return newnumber


def startemulator(env, classifier):
    done = True
    action = 0
    old_x_pos = 10000000
    for step in range(20000):
        time.sleep(0.03)
        if done or step == 0:
            state = env.reset()
            classifier = ImageClassifier.createimage_classifier(240, 256)
        state, reward, done, info = env.step(action)
        print(info['x_pos'])
        display = env.render('rgb_array')
        env.render()
        display = np.expand_dims(display, axis=0)
        action = classifier.predict(display)
        print('Action out of the Classifier')
        print(action)
        action = calculateaction(action, movements[2])
        print('Action after')
        print(action)
        if step % 120 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = 10000000
            else:
                old_x_pos = info['x_pos']
    env.close()


environment = createenvironment(environmentsmario[0], movements[2])
model = ImageClassifier.createimage_classifier(240, 256)

# ann_viz(model, title="My first neural network")

startemulator(environment, model)
