import numpy as np

# Sample for use when creating code to calculate actions
# [[0.02271274 0.04143925 0.03421655 0.76110446 0.35076535 0.18614534]]


def calculate_action_list(prediction):
    # TODO figure out how to calculate an action
    print(prediction)
    threshold = np.sum(prediction)/len(prediction)
    print(threshold)
    # threshold = 0.4  # (prediction[np.argmin(prediction)] + prediction[np.argmax(prediction)])/2
    action_list = np.zeros(prediction.shape)

    for num in range(len(prediction)):
        if prediction[num] > threshold:
            action_list[num] = 1
        else:
            action_list[num] = 0
    return action_list


def calculate_action_num(action_list):
    action_list = action_list.astype(int)
    print(action_list)
    action_dictionary = {
        '[0 0 0 0 0 0]': 0,      # 000000
        '[0 0 0 0 0 1]': 1,      # 000001
        '[0 0 0 0 1 0]': 2,      # 000010
        '[0 0 0 0 1 1]': 3,      # 000011
        '[0 0 0 1 0 0]': 4,      # 000100
        '[0 0 0 1 0 1]': 5,      # 000101
        '[0 0 0 1 1 0]': 6,      # 000110
        '[0 0 0 1 1 1]': 7,      # 000111
        '[0 0 1 0 0 0]': 8,      # 001000
        '[0 0 1 0 0 1]': 9,      # 001001
        '[0 0 1 0 1 0]': 10,     # 001010
        '[0 0 1 0 1 1]': 11,     # 001011
        '[0 1 0 0 0 0]': 12,     # 010000
        '[1 0 0 0 0 0]': 13      # 100000
    }
    return action_dictionary[str(action_list)]


COMPLEX_MOVEMENT = [
    ['NOP'],                # 000000
    ['B'],                  # 000001
    ['A'],                  # 000010
    ['A', 'B'],             # 000011
    ['right'],              # 000100
    ['right', 'B'],         # 000101
    ['right', 'A'],         # 000110
    ['right', 'A', 'B'],    # 000111
    ['left'],               # 001000
    ['left', 'B'],          # 001001
    ['left', 'A'],          # 001010
    ['left', 'A', 'B'],     # 001011
    ['down'],               # 010000
    ['up'],                 # 100000
]