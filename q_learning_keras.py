import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
from acions import NEW_COMPLEX_MOVEMENT

from Brain import Brain
import sys
import os
import cv2
movements = [NEW_COMPLEX_MOVEMENT, COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY]
movement = movements[3]
num_movement = len(movement)


load = False

brain = Brain()
brain.action_list = movement
if not load:
    brain.create_brain_q((60, 64))
else:
    brain.load_model('model_fitness_4208_generation_99_population_index_99.sav', os.getcwd() + "/models/")
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


num_episodes = 5000
best_fitness = 0
y = 0.95
eps = 0.7
decay_factor = 0.999
r_avg_list = []
env = create_environment(environments_mario[0], movement)
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    if eps < 0.1:
        eps = 0.1
    print("EPS value: " + str(eps))
    print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    step = 0
    old_x_pos = sys.maxsize
    image = env.render('rgb_array')
    image = cv2.resize(image, dsize=(64, 60), interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(image, axis=0)
    a = np.random.randint(0, num_movement)
    while not done:
        # env.render()
        if np.random.random() < eps:
            new_a = np.random.randint(1, num_movement)
        else:
            new_a = np.argmax(brain.model.predict(image))
        new_s, r, done, info = env.step(new_a)
        new_image = env.render('rgb_array')
        new_image = cv2.resize(new_image, dsize=(64, 60), interpolation=cv2.INTER_CUBIC)
        new_image = np.expand_dims(new_image, axis=0)
        if step % 60 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
            else:
                old_x_pos = info['x_pos']
        step += 1
        if info['time'] < 320 and r_sum < 300:
            done = True
        # target = r + y * np.max(brain.model.predict(new_image))
        if r < 1:
            target = 0
        else:
            target = 1
        target_vec = brain.model.predict(image)[0]
        print(target_vec)
        target_vec = np.zeros(num_movement)
        target_vec[a] = target
        print(target_vec)
        print("Action: " + str(a))
        print("Reward: " + str(r))
        brain.model.fit(image, target_vec.reshape(-1, num_movement), epochs=3, verbose=1)
        image = new_image
        s = new_s
        a = new_a   # swapping current action with the new action
        r_sum += r
        target_vec = None
    r_avg_list.append(r_sum)
    env.reset()
    # env.close()
    brain.fitness = r_sum
    print(str(i) + " FITNESS: " + str(r_sum))
    brain.save_model('models/')
    if best_fitness < r_sum:
        best_fitness = r_sum
    print("Current BEST: " + str(best_fitness))


print(fitness_list)
print(r_avg_list)
brain.save_model('models/')

