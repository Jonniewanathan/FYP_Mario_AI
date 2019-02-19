import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

from Brain import Brain
import sys
import os
import cv2

load = True

brain = Brain()
brain.action_list = SIMPLE_MOVEMENT
if not load:
    brain.create_brain_q((60, 64))
else:
    brain.load_model('model_fitness_1847_generation_99_population_index_99.sav', os.getcwd() + "/models/")
brain.generation_num = 99
brain.index = 99
fitness_list = []

environments_mario = ["SuperMarioBros-v0",
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
movement = movements[1]


def create_environment(_env, movement_set):
    environment = gym_super_mario_bros.make(_env)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movement_set)

    return environment


def run(model, env, iterations):
    for num in range(iterations):
        env.reset()
        step = 0
        action = 0
        old_x_pos = sys.maxsize
        fitness = []
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            fitness.append(reward)
            env.render()
            display = env.render('rgb_array')
            display = np.expand_dims(display, axis=0)
            action = np.argmax(brain.model.predict(display))
            # print(action)
            # print(info)
            # print(reward)
            if step % 120 == 0:
                if info['x_pos'] == old_x_pos:
                    done = True
                    old_x_pos = sys.maxsize
                else:
                    old_x_pos = info['x_pos']
            step += 1
        brain.fitness = sum(fitness)
    env.close()
    print("Fitness %d" % brain.fitness)
    fitness_list.append(brain.fitness)


num_episodes = 500

y = 0.99
eps = 0.5
decay_factor = 0.999
r_avg_list = []
for i in range(num_episodes):
    env = create_environment(environments_mario[0], movement)
    s = env.reset()
    eps *= decay_factor
    print("Episode {} of {}".format(i + 1, num_episodes))
    if i % 100 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    step = 0
    old_x_pos = sys.maxsize
    while not done:
        # env.render()
        image = env.render('rgb_array')
        image = cv2.resize(image, dsize=(64, 60), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, axis=0)
        if np.random.random() < eps:
            a = np.random.randint(0, 7)
        else:
            a = np.argmax(brain.model.predict(image))
        new_s, r, done, info = env.step(a)
        if step % 60 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
                r = -15
            else:
                old_x_pos = info['x_pos']
        step += 1
        target = r + y * np.max(brain.model.predict(image))
        target_vec = brain.model.predict(image)[0]
        target_vec[a] = target
        print(target_vec)
        brain.model.fit(image, target_vec.reshape(-1, 7), epochs=1, verbose=1)
        s = new_s
        r_sum += r
        target_vec = None
    r_avg_list.append(r_sum)
    env.close()
    if brain.fitness < r_sum:
        brain.fitness = r_sum
        brain.save_model('models/')
    else:
        brain.fitness = r_sum


print(fitness_list)
print(r_avg_list)
brain.save_model('models/')

