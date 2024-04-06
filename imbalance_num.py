import random


def imbalance(k):
    random.seed(k)
    return [random.randint(1, 300) for _ in range(22)]