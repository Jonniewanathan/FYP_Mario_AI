from Image_Classifier import ImageClassifier
import numpy as np
import gym_super_mario_bros
import multiprocessing as mp
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY


class Env(object):
    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.done = False

    def calculateaction(self, number, action):
        new_number = float(number[0]) * len(action)
        # print(newnumber)
        new_number = round(new_number) - 1

        if new_number < 0:
            new_number = 0

        return new_number

    def run(self):
        classifier = ImageClassifier.createimage_classifier(240, 256)
        action = 0
        old_x_pos = 10000000
        fitness = []
        best_fitness = 0
        best_brain = classifier
        for step in range(5000):
            if self.done:
                print("Fitness %d" % sum(fitness))
                if best_fitness < sum(fitness):
                    best_fitness = sum(fitness)
                    best_brain = classifier
                state = self.env.reset()
                self.done = False
                classifier = ImageClassifier.createimage_classifier(240, 256)
                fitness = []
            state, reward, self.done, info = self.env.step(action)
            fitness.append(reward)
            self.env.render()
            display = self.env.render('rgb_array')
            display = np.expand_dims(display, axis=0)
            action = classifier.predict(display)
            action = self.calculateaction(action, RIGHT_ONLY)
            print(action)
            print(info)
            print(reward)
            if step % 120 == 0:
                if info['x_pos'] == old_x_pos:
                    self.done = True
                    old_x_pos = 10000000
                else:
                    old_x_pos = info['x_pos']

        self.env.close()
        print(classifier.get_config())
        print(best_fitness)

        best_brain.save("C:/models/best_model_fitness_" + str(best_fitness) + ".sav", overwrite=True)


def runner():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
    env = Env(env)
    env.run()


if __name__ == '__main__':
    procs = []

    for idx in range(1):
        proc = mp.Process(target=runner)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
