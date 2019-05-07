from multiprocessing import Pool
from multiprocessing import Process
import time
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


def createenvironment(enviro, movementset):
    environment = gym_super_mario_bros.make(enviro)
    environment = BinarySpaceToDiscreteSpaceEnv(environment, movementset)

    return environment


def startemulator(env):
    done = True
    old_x_pos = 10000000
    for step in range(5000):
        time.sleep(0.06)
        if done or step == 0:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        print(info)
        print(step)
        # env.render()
        if step % 60 == 0:
            if info['x_pos'] == old_x_pos:
                done = True
                old_x_pos = 10000000
            else:
                old_x_pos = info['x_pos']
    env.close()


def start_pool():
    environments = []
    for num in range(4):
        environments.append(createenvironment(environmentsmario[0], movements[2]))

    pool = Pool()
    pool.map(startemulator, environments)
    # pool.close()
    # pool.join()


def start_pool_new():
    environments = []
    for num in range(4):
        environments.append(createenvironment(environmentsmario[0], movements[2]))
    with Pool(5) as p:
        p.map(startemulator, environments)


def start_process_new():
    environments = []
    for num in range(4):
        environments.append(createenvironment(environmentsmario[0], movements[2]))
    for env in environments:
        p = Process(target=startemulator, args=(env,))
        p.start()
        p.join()


def start_simple_emulator():
    startemulator(createenvironment(environmentsmario[0], movements[2]))


if __name__ == '__main__':
    start_process_new()
    # start_simple_emulator()
    # for number in range(4):
    #     print('run number' + str(number))
    #     start_simple_emulator()
