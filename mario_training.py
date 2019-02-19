import sys
import random
import os
from NEAT import NEAT
import cv2
import numpy as np
import gym_super_mario_bros
import multiprocessing as mp
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

movement_sets = [COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY]

movement_set = movement_sets[0]
world_num = 1
level_num = 1

num_generations = 1

level = "SuperMarioBros-" + str(world_num) + "-" + str(level_num) + "-v0"

num_population = 30
num_threads = 2
render_probability = 0

generation = NEAT.create_random_generation(num_population, movement_set)
generation_copy = []
environment_list = []


class Env(object):
    def __init__(self, env, brain, render):
        self.brain = brain
        self.render = render
        self.env = env
        self.env.reset()
        self.done = False
        self.brain.action_list = movement_set

    def run(self):
        # self.env.reset()
        step = 0
        action = 0
        old_x_pos = sys.maxsize
        fitness = []
        while not self.done:
            state, reward, self.done, info = self.env.step(action)
            fitness.append(reward)
            if self.render:
                self.env.render()
            image = self.env.render('rgb_array')
            image = cv2.resize(image, dsize=(32, 30), interpolation=cv2.INTER_CUBIC)
            image = np.expand_dims(image, axis=0)
            action = np.argmax(self.brain.model.predict(image))
            print("Action: " + str(action))
            print(info)
            print("Reward: " + str(reward))
            if step % 120 == 0:
                if info['x_pos'] == old_x_pos:
                    self.done = True
                    old_x_pos = sys.maxsize
                else:
                    old_x_pos = info['x_pos']
            step += 1
        self.brain.fitness = sum(fitness)
        self.env.close()
        print("Fitness %d" % sum(fitness))


def runner(proc_num, split_environment):
    for env in split_environment:
        print("--------------environments Left: " + str(len(split_environment)) + " Proc_num " + str(proc_num) +  "------------------------------")
        split_environment.pop().run()
    generation_copy.sort(key=lambda x: x.fitness)
    # for _brain in generation_copy:
    #     print(_brain.fitness)
    best_brain = generation_copy.pop()
    print(best_brain.fitness)
    # if os.path.isdir(os.getcwd() + "/models"):
    #     best_brain.save_model(os.getcwd() + "/models/")
    #     print(os.getcwd())
    # else:
    #     os.makedirs(os.getcwd() + "/models")
    #     print(os.getcwd())
    #     best_brain.save_model(os.getcwd() + "/models/")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    if n > 1:
        l = list(l)
        for i in range(0, len(l), n):
            yield l[i:i + n]
    else:
        return list(l)


for num in range(len(generation)):
    mario_brain = generation.pop()
    if random.random() < render_probability:
        _render = True
    else:
        _render = False
    environment = gym_super_mario_bros.make(level)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movement_set)
    environment = Env(environment, mario_brain, _render)
    environment_list.append(environment)
    generation_copy.append(mario_brain)

if __name__ == '__main__':

    procs = []
    print(environment_list)
    split_environments = chunks(environment_list, num_threads)
    split_environments = list(split_environments)

    for idx in range(num_threads):
        proc = mp.Process(target=runner, args=(idx, split_environments[idx]))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

