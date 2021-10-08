import numpy as np
from numpy import random


def random_argmax(value_list):
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


class Agent:

    def __init__(self, stand, stats, epsilon=0.1):
        self.stand = stand
        self.stats = stats
        self.epsilon = epsilon

    def run(self, num_exps, num_steps):

        for e in range(num_exps):
            self.stand.initialize()
            for s in range(num_steps):
                arm = self.__pull()
                reward = arm.reward()

                self.stats.record(e, s, arm.num, reward)

    def rewards_matrix(self):
        return self.rewards_matrix

    def __pull(self):
        p = random.random()

        if p < self.epsilon:
            arm = random.choice(self.stand.arms)
        else:
            arm = self.stand.arms[random_argmax([arm.reward() for arm in self.stand.arms])]

        return arm
