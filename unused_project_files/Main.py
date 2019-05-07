import sys
import random
import os
from NEAT import NEAT
from collections import deque
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

level = "SuperMarioBros-" + str(world_num) + "-" + str(level_num) + "-v0"

num_generations = 1
num_population = 20
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
        self.env.reset()
        step = 0
        action = 0
        old_x_pos = sys.maxsize
        fitness = []
        while not self.done:
            state, reward, self.done, info = self.env.step(action)
            fitness.append(reward)
            if self.render:
                self.env.render()
            display = self.env.render('rgb_array')
            display = np.expand_dims(display, axis=0)
            action = self.brain.calculate_action(display)
            # print(action)
            # print(info)
            # print(reward)
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


def runner(proc_num, environments_list):
    for num in range(num_population):
        env = environments_list.pop()
        print("--------------environments Left: " + str(len(environments_list)) + "------------------------------")
        env.run()
    generation_copy.sort(key=lambda x: x.fitness)
    # for _brain in generation_copy:
    #     print(_brain.fitness)
    best_brain = generation_copy.pop()
    print(best_brain.fitness)
    # return_dict[proc_num] = best_brain
    # print(return_dict)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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

    for num_generation in range(num_generations):
        print("--------------Generation Number: " + str(num_generation) + " START ------------------------------")
        procs = []
        manager = mp.Manager()
        return_dict = manager.dict()
        split_environment = list(chunks(environment_list, num_threads))
        print(split_environment)
        print(split_environment[0])

        for idx in range(num_threads):
            proc = mp.Process(target=runner, args=(idx, deque(split_environment[idx])))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
        print("printing best brain fitness")
        # print(return_dict[0].fitness)
        # print("generation number: " + str(return_dict[0].generation_num))

        if os.path.isdir(os.getcwd() + "/models"):
            return_dict[0].save_model(os.getcwd() + "/models/")
            print(os.getcwd())
        else:
            os.makedirs(os.getcwd() + "/models")
            print(os.getcwd())
            return_dict[0].save_model(os.getcwd() + "/models/")

        # if num_generation != num_generations-1:
        #     print("Just before Generation Creation")
        #     generation = NEAT.create_generation_from_model(return_dict[0], num_population, num_generation + 1)
        #     print("Just after generation creation")
        #     for num in generation:
        #         mario_brain = generation.pop()
        #         if random.random() < render_probability:
        #             _render = True
        #         else:
        #             _render = False
        #         environment = gym_super_mario_bros.make(level)
        #         environment = BinarySpaceToDiscreteSpaceEnv(environment, movement_set)
        #         environment = Env(environment, mario_brain, _render)
        #         environment_list.append(environment)
        #         generation_copy.append(mario_brain)
        #     print("--------------Generation Number: " + str(num_generation) + " END------------------------------")


