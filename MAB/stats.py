import numpy as np


class Stats:

    def __init__(self, arms, num_experiments, num_steps):
        self.optimal_n = np.zeros(num_experiments)
        self.optimal_arm_indices = np.argwhere(
            [arm.q for arm in arms] == np.max([arm.q for arm in arms])).flatten().tolist()
        self.total_reward = np.zeros(shape=(num_experiments, num_steps))
        self.mean_reward = np.zeros(shape=(num_experiments, num_steps))
        self.rewards = np.zeros(shape=(num_experiments, num_steps, len(arms)))
        self.optimal_action_percent = np.zeros(shape=(num_experiments, num_steps))

    def record(self, e, s, arm_index, R):
        self.total_reward[e][s] = self.total_reward[e][s - 1] + R if s > 0 else R
        self.mean_reward[e][s] = self.total_reward[e][s] / (s + 1)
        self.rewards[e][s][arm_index] = R
        if arm_index in self.optimal_arm_indices:
            self.optimal_n[e] += 1

        self.optimal_action_percent[e][s] = self.optimal_n[e] / (s + 1)
