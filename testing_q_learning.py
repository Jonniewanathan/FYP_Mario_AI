import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

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


def naive_sum_reward_agent(env, num_episodes=500):
    # this is the table that will hold our summated rewards for
    # each action in each state
    r_table = np.zeros((10000, len(movement)))
    for g in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            if np.sum(r_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, len(movement))
                if a > len(movement):
                    a = np.random.randint(0, len(movement))
            else:
                # select the action with highest cummulative reward
                a = np.argmax(r_table[s, :])
                if a > len(movement): # needed for an odd issue where a huge number is generated as an action
                    a = np.random.randint(0, len(movement))
                # state, reward, done, info
            new_s, r, done, info = env.step(a)
            r_table[s, a] += r
            s = new_s
    return r_table


def q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((10000, len(movement)))
    y = 0.95
    lr = 0.8
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            if np.sum(q_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, len(movement))
                if a > len(movement):  # needed for an odd issue where a huge number is generated as an action
                    a = np.random.randint(0, len(movement))
            else:
                # select the action with largest q value in state s
                a = np.argmax(q_table[s, :])
                if a > len(movement):  # needed for an odd issue where a huge number is generated as an action
                    a = np.random.randint(0, len(movement))
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr*(y*np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table


def eps_greedy_q_learning_with_table(env, num_episodes=10):
    q_table = np.zeros((10000, len(movement)))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            env.render()
            image = env.render('rgb_array')
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[image, :]) == 0:
                a = np.random.randint(0, len(movement))
            else:
                a = np.argmax(q_table[image, :])
                if a > len(movement):  # needed for an odd issue where a huge number is generated as an action
                    a = np.random.randint(0, len(movement))
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            q_table[image, a] += r + lr * (y * np.max(q_table[image, :]) - q_table[image, a])
            s = new_s
    return q_table


def test_methods(env, num_iterations=2):
    winner = np.zeros((3,))
    for g in range(num_iterations):
        m0_table = naive_sum_reward_agent(env, 5)
        m1_table = q_learning_with_table(env, 5)
        m2_table = eps_greedy_q_learning_with_table(env, 5)
        m0 = run_game(m0_table, env)
        m1 = run_game(m1_table, env)
        m2 = run_game(m2_table, env)
        w = np.argmax(np.array([m0, m1, m2]))
        winner[w] += 1
        print("Game {} of {}".format(g + 1, num_iterations))
    return winner


def run_game(table, env):
    s = env.reset()
    tot_reward = 0
    done = False
    while not done:
        a = np.argmax(table[s, :])
        if a > len(movement):  # needed for an odd issue where a huge number is generated as an action
            a = np.random.randint(0, len(movement))
        s, r, done, _ = env.step(a)
        tot_reward += r
    return tot_reward


def create_environment(enviro, movement_set):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movement_set)

    return environment


if __name__ == "__main__":
    print(test_methods(create_environment(environments_mario[0], movements[0])))

