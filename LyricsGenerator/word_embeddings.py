import numpy as np
import math


def word2vec(word):
    with open("data/glove.6B.100d.txt", 'r') as glove:
        for line in glove:
            words = line.split(" ")
            if words[0] == word:
                words.pop(0)
                return np.array(list(map(float, words)))
    return np.random.randn(100)


def dist(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)


def vec2word(vec):
    with open("data/glove.6B.100d.txt", 'r') as glove:
        closest_word = None
        closest_distance = math.inf
        for line in glove:
            words = line.split(" ")
            current_word = words[0]
            words.pop(0)
            current_distance = dist(np.array(list(map(float, words))), vec)
            if current_distance < closest_distance:
                closest_distance = current_distance
                closest_word = current_word
            if closest_distance == 0:
                break
        return closest_word

