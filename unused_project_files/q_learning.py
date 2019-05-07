import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

from keras.utils import plot_model
import sys

from Brain import Brain

brain = Brain()
brain.action_list = SIMPLE_MOVEMENT
brain.create_random_brain((240, 256))


# TODO ADD in configuration variables here

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


def eps_greedy_q_learning_with_table(env, num_episodes=5):
    q_table = np.zeros((5000, len(movement)))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    step = 0
    old_x_pos = sys.maxsize
    actions_np = np.zeros((1, 7))
    # actions = []
    # images = []
    # images_np = np.ndarray((1, 240, 256, 3), dtype=np.uint8)
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        index = 0
        while not done:
            # env.render()
            image = env.render('rgb_array')
            image = np.expand_dims(image, axis=0)
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[index, :]) == 0:
                # a = np.argmax(brain.model.predict(image))
                a = np.random.randint(0, len(movement))
            else:
                a = np.argmax(q_table[index, :])
                if a > len(movement):  # needed for an odd issue where a huge number is generated as an action
                    a = np.argmax(brain.model.predict(image))
            # pdb.set_trace()
            new_s, r, done, info = env.step(a)
            if step % 120 == 0:
                if info['x_pos'] == old_x_pos:
                    done = True
                    old_x_pos = sys.maxsize
                else:
                    old_x_pos = info['x_pos']
            step += 1
            q_table[index, a] += r + lr * (y * np.max(q_table[index, :]) - q_table[index, a])
            s = new_s
            # actions.append([0, 0, 0, 0, 0, 0, 0])
            # print(actions)
            # actions[index][int(np.argmax(q_table[index]))] = 1
            actions_np[0][int(np.argmax(q_table[index]))] = 1
            # actions_np = np.array(actions)
            actions_np[0][np.argmax(q_table[index])] = 1
            # images.append(image)
            # images_np[0] = np.array(images)
            # print(actions_np[index])
            brain.model.fit(x=image, y=actions_np, epochs=1, batch_size=1, verbose=1)
            actions_np = np.zeros((1, 7))
            # brain.model.fit(x=images_np.reshape((len(images), 240, 256, 3)), y=actions_np, epochs=1, batch_size=1, verbose=1)
            index += 1
    return q_table


def create_environment(_env, movement_set):
    environment = gym_super_mario_bros.make(_env)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movement_set)

    return environment


def run(model, env):
    for num in range(5):
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
            action = np.argmax(model.predict(display))
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


if __name__ == "__main__":
    # lot_model(brain.model, to_file='models/model.png')

    env = create_environment(environments_mario[0], movement)
    print(brain.model.layers[0].get_config())

    final_q_table = eps_greedy_q_learning_with_table(env, 5)

    run(brain.model, env)

    print(final_q_table)

