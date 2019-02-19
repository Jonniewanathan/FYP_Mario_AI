from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from acions import COMPLEX_MOVEMENT
import acions
import os
import pandas as pd

# train_dir = os.getcwd() + "/capture_2019-02-171550446316.532347/"
# csv_train = pd.read_csv(train_dir + "controller_capture.csv")
# csv_train = csv_train.drop(csv_train.columns[0], axis=1)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

# array = csv_train.get_values()

done = True
action = 0
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(7)
    # counter = 400 - info['time']
    # action = acions.calculate_action_num(array[counter + 14])
    # print(counter)
    env.render()
env.close()

