from Image_Classifier import ImageClassifier
from Brain import Brain
from NEAT import NEAT
import keras

Brain = Brain()


def print_model(model):
    for weight in model.get_weights():
        print(weight)
    print("This is the shape of the weights lists")
    for weight in model.get_weights():
        print(weight.shape)
    print(model.get_weights())
    for layer in model.layers:
        print(layer)

# classifier = keras.models.load_model('C:/models/best_model_fitness_3108.sav')


# classifier = ImageClassifier.createimage_classifier(240, 256)
# Brain.create_random_brain([240, 256], 1, 1, 1)

Brain.create_random_brain([240, 256])
# new_brain = Brain.copy()

# generation = NEAT.create_random_generation(100)
#
# for brain in generation:
#     print(str(brain.generation_num) + "_" + str(brain.index) + "_" + str(brain.fitness))
#     print_model(brain.model)


print_model(Brain.model)
# print_model(new_brain.model)

# print_model(classifier)
# print_model(Brain.model)

# print("Loaded Model")
# print_model(classifier)

# for num in range(2):
#     Brain.create_random_brain([240, 256])
#     print("Model %d" % num)
#     print_model(Brain.model)
