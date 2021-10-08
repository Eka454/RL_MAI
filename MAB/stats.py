import numpy as np


class Stats:

    def __init__(self, stand, num_experiments, num_steps):
        self.stand = stand
        self.optimal_n = np.zeros(num_experiments)
        self.optimal_arm_indices = np.argwhere([arm.q for arm in stand.arms] == np.max([arm.q for arm in stand.arms])).flatten().tolist()
        self.total_reward = np.zeros(shape=(num_experiments, num_steps))
        self.rewards = np.zeros(shape=(num_experiments, num_steps, len(self.stand.arms)))
        self.optimal_action_percent = np.zeros(shape=(num_experiments, num_steps))

    def record(self, exp_num, step_num, arm_index, R):
        self.total_reward[exp_num][step_num] = self.total_reward[exp_num].sum()
        self.rewards[exp_num][step_num][arm_index] = R
        if arm_index in self.optimal_arm_indices:
            self.optimal_n[exp_num] += 1

        self.optimal_action_percent[exp_num][step_num] = self.optimal_n[exp_num] / step_num
