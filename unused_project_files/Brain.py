from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation
import keras
import random
from keras.optimizers import SGD, Adam
import numpy as np


class Brain:
    def __init__(self, model=None, layers_nodes=None):

        if model is not None:
            self.model = model
        if layers_nodes is not None:
            self.layers_nodes = layers_nodes
        self.fitness = 0
        self.generation_num = 0
        self.index = 0
        self.action_list = None

    def create_random_brain(self, image_shape):
        input_shape = (image_shape[0], image_shape[1], 3)

        self.create_random_layers_nodes('Mario')

        self.model = Sequential()
        # Input
        self.add_conv2d_layer(1, 'relu', input_shape)
        # Hidden
        # self.model.add(Dense(activation="sigmoid", units=1))
        self.add_dense_layer(1, 'relu')
        for num in self.layers_nodes:
            self.add_dense_layer(num, 'relu')
        # Output
        # self.model.add(Dense(activation="sigmoid", units=7))
        self.add_dense_layer(len(self.action_list), 'softmax')
        # compiling the model
        self.compile_model()

    def create_brain_q(self, image_shape):
        input_shape = (image_shape[0], image_shape[1], 3)

        self.model = Sequential()
        # Input
        self.model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(len(self.action_list), activation='softmax'))
        # sgd = SGD(lr=0.09, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])

    def create_conv_q_brain(self, input_shape):
        optimizer = Adam(lr=0.001)
        objective = 'binary_crossentropy'

        self.model = Sequential()

        self.model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
        self.model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        self.model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        self.model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
        self.model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(len(self.action_list)))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    def create_brain(self, image_shape):
        input_shape = (image_shape[0], image_shape[1], 3)

        self.model = Sequential()
        # Input
        self.add_conv2d_layer(28, 'relu', input_shape)
        # Hidden
        for num in self.layers_nodes:
            self.add_dense_layer(self.layers_nodes[num], 'relu')
        # Output
        self.add_dense_layer(len(self.action_list), 'sigmoid')
        self.compile_model()

    def create_brain_ga(self):
        self.model = Sequential()
        self.add_conv2d_layer(64, 'relu', (240, 256, 3))
        self.model.add(Dense(output_dim=7, input_dim=3, activation="sigmoid"))
        self.model.add(Dense(output_dim=1, activation="sigmoid"))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])

    def create_random_layers_nodes(self, seed):
        self.layers_nodes = []
        random.seed(seed)
        length = random.randint(1, 5)
        for num in range(length):
            self.layers_nodes.append(random.randint(1, 16))

    def add_dense_layer(self, nodes, activation):
        self.model.add(Dense(nodes, activation=activation))

    def add_conv2d_layer(self, nodes, activation, input_shape):
        self.model.add(Conv2D(nodes, (3, 3), input_shape=input_shape, activation=activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        # self.model.add(Dropout(0.2))

    def compile_model(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])

    # def visualise(self):
    #     # TODO create a way to visualise the neural network
    #

    def calculate_action(self, image):
        new_number = self.model.predict(image)
        new_number = new_number[0]
        # TODO Create function to calculate the action based on the image input returns an int for one action
        new_number = new_number[0] * len(self.action_list)
        # print(new_number)
        new_number = round(new_number) - 1

        if new_number < 0:
            new_number = 0

        return new_number

    def calculate_action_train(self, image, reward):
        numpy_reward = np.array([reward])
        new_number = self.model.predict(image)
        self.model.fit(x=image, y=numpy_reward, epochs=4, verbose=1)
        new_number = new_number[0]
        new_number = new_number[0] * len(self.action_list)
        new_number = round(new_number) - 1

        if new_number < 0:
            new_number = 0

        return new_number

    # Save a model of the a specific name and to a specific directory
    def save_model(self, directory):
        self.model.save(directory + "model_fitness_" + str(self.fitness) + "_generation_" + str(self.generation_num) +
                        "_population_index_" + str(self.index) + ".sav", overwrite=True)

    # Loads a model from a directory and filename
    def load_model(self, file_name, directory):
        self.model = keras.models.load_model(directory + file_name)

    def copy(self):
        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        return Brain(model_copy)
