from Brain import Brain
import numpy as np
import random


class NEAT:

    @staticmethod
    def create_random_generation(num_population, action_list):
        generation = []
        for pop_index in range(num_population):
            brain = Brain()
            brain.action_list = action_list
            brain.create_random_brain([30, 32])
            brain.index = pop_index
            generation.append(brain)

        return generation

    @staticmethod
    def create_generation_q_learning(num_population):
        generation = []
        for pop_index in range(num_population):
            brain = Brain()
            brain.create_brain_q([30, 32])
            brain.index = pop_index
            generation.append(brain)

        return generation

        # TODO create function to create a generation of Brains

    @staticmethod
    def create_generation_from_model(brain, num_population, num_generation):
        generation = []
        for pop_index in range(num_population):
            new_brain = brain.copy()
            new_brain.index = pop_index
            new_brain.generation_num = num_generation

            generation.append(brain)

        return generation

    #     # TODO make a function that will mutate a model and return a new model from that
    #
    # @staticmethod
    # def crossover(self, brain1, brain2):
    #     # TODO create a function that takes in 2 brains and returns a new brain based on those taken in
