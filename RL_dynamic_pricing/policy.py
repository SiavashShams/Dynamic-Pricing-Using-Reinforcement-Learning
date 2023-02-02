import numpy as np
import random


class PricingPolicy:
    def __init__(self, env, alpha=0.1, discount_rate=0.99):
        self.env = env
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.num_actions = 120
        self.q_table = np.zeros((self.env.demand_level_max - self.env.demand_level_min + 1, self.num_actions))

    def select_action(self, state):
        demand_level, _, _ = state
        if random.uniform(0, 1) < 0.1:
            # Select a random action with probability epsilon
            action = random.choice(range(self.num_actions))
        else:
            # Select the action with the highest Q-value with probability (1 - epsilon)
            action = np.argmax(self.q_table[demand_level - self.env.demand_level_min])
            print(action)
        return action

    def update(self, state, action, reward):
        demand_level, _, _ = state
        alpha, discount_rate = self.alpha, self.discount_rate
        q_old = self.q_table[demand_level - self.env.demand_level_min, action]
        q_new = q_old + alpha * (
                    reward + discount_rate * np.max(self.q_table[demand_level - self.env.demand_level_min]) - q_old)
        self.q_table[demand_level - self.env.demand_level_min, action] = q_new
