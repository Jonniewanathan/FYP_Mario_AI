import sys
import random
import os
import cv2
from NEAT import NEAT
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
num_population = 1
num_threads = 1
render_probability = 1

generation = NEAT.create_generation_q_learning(num_population)
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
        num_episodes = 200

        y = 0.99
        eps = 0.5
        decay_factor = 0.999
        r_avg_list = []
        self.env.reset()
        for i in range(num_episodes):
            # env = create_environment(environments_mario[0], movement)
            s = self.env.reset()
            eps *= decay_factor
            print("Episode {} of {}".format(i + 1, num_episodes))
            done = False
            r_sum = 0
            step = 0
            old_x_pos = sys.maxsize
            while not done:
                if i % 10 == 0:
                    self.env.render()
                image = self.env.render('rgb_array')
                image = cv2.resize(image, dsize=(32, 30), interpolation=cv2.INTER_CUBIC)
                image = np.expand_dims(image, axis=0)
                prediction = self.brain.model.predict(image, verbose=1)
                if np.random.random() < eps:
                    a = np.random.randint(0, 12)
                else:
                    a = np.argmax(prediction)
                # a = np.argmax(prediction)
                new_s, r, done, info = self.env.step(a)
                if step % 240 == 0:
                    if info['x_pos'] == old_x_pos:
                        done = True
                        old_x_pos = sys.maxsize
                        # r = -15
                    else:
                        old_x_pos = info['x_pos']
                step += 1
                target = r + y * np.max(prediction)
                print("Reward" + str(r))
                print("Max Prediction" + str(np.max(prediction)))
                print("action Prediction" + str(np.argmax(prediction)))
                target_vec = np.copy(prediction[0])
                target_vec[a] = target
                # print(target_vec)
                print("predict")
                print(prediction)
                self.brain.model.fit(image, target_vec.reshape(-1, 12), epochs=1, verbose=1)
                print("target_vec")
                print(target_vec.reshape(-1, 12))
                s = new_s
                r_sum += r
            r_avg_list.append(r_sum)
            # self.env.close()
            if self.brain.fitness < r_sum and i != 0:
                self.brain.fitness = r_sum
                self.brain.save_model('models/')
            else:
                self.brain.fitness = r_sum
        print(r_avg_list)
        self.brain.save_model('models/')
        self.env.close()


def runner():
    for num in range(num_population):
        env = environment_list.pop()
        print("--------------environments Left: " + str(len(environment_list)) + "------------------------------")
        env.run()


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

        for idx in range(num_threads):
            proc = mp.Process(target=runner)
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()


