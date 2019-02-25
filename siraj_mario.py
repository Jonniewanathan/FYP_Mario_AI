# INITIALIZATION: libraries, parameters, network...

from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten, Conv2D, Activation   # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque               # For storing moves
import numpy as np
import random                             # For sampling batches from the observations
import sys
import os

# Gym Imports
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

movement_set = movements[2]
environment = environmentsmario[0]

num_actions = len(movement_set)

world_num = 1
level_num = 1

level = "SuperMarioBros-" + str(world_num) + "-" + str(level_num) + "-v0"


def create_environment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


env = create_environment(level, movement_set)

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
# model = Sequential()
# model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
# model.add(Flatten())       # Flatten input so as to have no problems with processing
# model.add(Dense(18, init='uniform', activation='relu'))
# model.add(Dense(10, init='uniform', activation='relu'))
# model.add(Dense(num_actions, init='uniform', activation='linear'))   # Same number of outputs as possible actions
#
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


cnn_input = env.observation_space.shape
model = Sequential()
model.add(Conv2D(32, (8, 8),  input_shape=cnn_input, strides=(4, 4)))
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_actions))
model.add(Activation('softmax'))
# build the mask using the functional API

# put the two pieces of the graph together

# build the model
# compile the model with the default loss and optimization technique
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# Parameters
D = deque()                                # Register where the actions will be stored

observe_time = 500                          # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7                              # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 50
episodes = 100
best_reward = 0
reward_list = []

for eps in range(episodes):

    print("episode: " + str(eps) + "of" + str(episodes))

    # FIRST STEP: Knowing what each action does (Observing)

    observation = env.reset()                     # Game begins
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    state = np.stack((obs, obs), axis=1)
    done = False
    old_x_pos = sys.maxsize
    for t in range(observe_time):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, num_actions)
        else:
            Q = model.predict(state[0])          # Q-values predictions
            action = np.argmax(Q)             # Move with highest Q-value is the chosen one
        observation_new, reward, done, info = env.step(action)    # See state of the game, reward... after performing the action
        if t % 240 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
                r = -15
            else:
                old_x_pos = info['x_pos']
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new         # Update state
        if done:
            env.reset()           # Restart game if it's finished
            obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
            state = np.stack((obs, obs), axis=1)
    print('Observing Finished')

    # SECOND STEP: Learning from the observations (Experience replay)

    minibatch = random.sample(D, mb_size)  # Sample some moves

    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, num_actions))

    for i in range(0, mb_size):
        print("Training " + str(i) + "of" + str(mb_size))
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

        # Build Bellman equation for the Q function
        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state[0])
        Q_sa = model.predict(state_new[0])

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

        # Train network to output the Q function
        model.train_on_batch(inputs, targets)
    print('Learning Finished')

    # THIRD STEP: Play!

    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    old_x_pos = sys.maxsize
    step = 0
    while not done:
        env.render()                    # Uncomment to see game running
        Q = model.predict(state[0])
        action = np.argmax(Q)
        observation, reward, done, info = env.step(int(action))
        if step % 240 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
                # r = -15
            else:
                old_x_pos = info['x_pos']
        step += 1
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))
    if tot_reward > best_reward:
        best_reward = tot_reward
        model.save(os.getcwd() + "/models/model_" + str(tot_reward), True, True)
    reward_list.append(tot_reward)

print(reward_list)







