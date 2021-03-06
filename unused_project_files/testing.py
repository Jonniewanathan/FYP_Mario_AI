from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
import tensorflow as tf
from collections import deque               # For storing moves
import sys
import os
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
import random

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

movement_set = movements[0]
environment = environmentsmario[0]
                            # For sampling batches from the observations


def create_environment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


env = create_environment(environment, movement_set)

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.

model = tf.keras.models.Sequential()
model.add((tf.keras.layers.Dense(20, input_shape=(2,) + env.observation_space.shape)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Flatten())
model.add((tf.keras.layers.Dense(18)))
model.add(tf.keras.layers.Activation('relu'))
model.add((tf.keras.layers.Dense(32)))
model.add(tf.keras.layers.Activation('relu'))
model.add((tf.keras.layers.Dense(10)))
model.add(tf.keras.layers.Activation('relu'))
model.add((tf.keras.layers.Dense(env.action_space.n)))
model.add(tf.keras.layers.Activation('linear'))   # Same number of outputs as possible actions

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    loss=tf.keras.losses.mse,
    metrics=['accuracy']
)

# tpu_model = tf.contrib.tpu.keras_to_tpu_model(
#     model,
#     strategy=tf.contrib.tpu.TPUDistributionStrategy(
#         tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR']), True
#     )
# )

# tpu_model.compile(
#     optimizer=tf.train.AdamOptimizer(),
#     loss=tf.keras.losses.mse,
#     metrics=['accuracy']
# )

# Parameters
D = deque()                                # Register where the actions will be stored

observe_time = 600                          # Number of timesteps we will be acting on the game and observing results
epsilon = 0.5                              # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 50
episodes = 200
best_reward = 0
best_model = None
reward_list = []
best_models = []


for eps in range(episodes):
    D = deque()
    print("episode " + str(eps) + " of " + str(episodes))

    # FIRST STEP: Knowing what each action does (Observing)

    observation = env.reset()                     # Game begins
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    state = np.stack((obs, obs), axis=1)
    done = False
    old_x_pos = sys.maxsize
    for t in range(observe_time):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=8)[0]
        else:
            Q = model.predict(state)          # Q-values predictions
            action = np.argmax(Q)             # Move with highest Q-value is the chosen one
        observation_new, reward, done, info = env.step(action)    # See state of the game, reward... after performing the action
        if t % 240 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = sys.maxsize
                # r = -15
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
    targets = np.zeros((mb_size, env.action_space.n))

    for i in range(0, mb_size):
        print("Training " + str(i) + "of" + str(mb_size))
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

        # Build Bellman equation for the Q function
        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

        # Train network to output the Q function
        model.train_on_batch(inputs, targets)
        # model.train_on_batch(inputs, targets)
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
        Q = model.predict(state)
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
    reward_list.append(tot_reward)
    if tot_reward > best_reward:
        best_reward = tot_reward
        # best_models.append(model)
        best_model = model
        model.save(os.getcwd() + "/models/model_" + str(tot_reward), True, True)

print(reward_list)