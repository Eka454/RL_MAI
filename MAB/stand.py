from numpy import random


class Arm:

    def __init__(self, num, q, Q):
        self.num = num
        self.q = q
        self.Q = Q
        self.num_of_pull = 0

    def reward(self):
        val = self.q + random.randn()
        self.__update_estimate(val)

        return 0 if val < 0 else val

    def estimate(self):
        return self.Q

    def __update_estimate(self, R):
        self.num_of_pull += 1
        self.Q = self.Q + (1.0 / self.num_of_pull) * (R - self.Q)


class Stand:

    def __init__(self, Q, rewards):
        self.Q = Q
        self.rewards = rewards
        self.initialize()

    def initialize(self):
        self.arms = [Arm(i, q, self.Q) for i, q in enumerate(self.rewards)]
