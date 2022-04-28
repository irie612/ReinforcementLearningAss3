#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class Agent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon, n_planning_updates):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_updates = n_planning_updates
        self.Q = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        self.r_function = lambda s, a, s_next: self.R_sum[s, a, s_next] / self.n[s, a, s_next]

    def select_action(self, s):
        p = np.random.uniform()
        if p <= 1 - self.epsilon:
            choices = np.argwhere(self.Q[s] == np.max(self.Q[s])).flatten()
            if choices.size==0:
                print(self.Q[s])
            a = np.random.choice(choices)
        else:
            choices = np.argwhere(self.Q[s] != np.max(self.Q[s])).flatten()
            if choices.size > 0:
                a = np.random.choice(choices)
            else:
                a = np.random.randint(self.Q[s].size)
        return a

    def update(self, s, a, r, s_next, done):
        self.n[s, a, s_next] += 1
        self.R_sum[s, a, s_next] += r

    def simulate_model(self, s, a):
        p_hat = self.n[s, a] / np.sum(self.n[s, a])
        s_next = np.random.choice([i for i in range(self.n_states)], p=p_hat)
        return s_next, self.r_function(s, a, s_next)


class DynaAgent(Agent):

    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon, n_planning_updates):
        super().__init__(n_states, n_actions, learning_rate, gamma, epsilon, n_planning_updates)

    def update(self, s, a, r, s_next, done):
        # TO DO: Add own code
        super().update(s, a, r, s_next, done)
        self.Q[s, a] += self.learning_rate * (r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])
        
        n_s = np.sum(self.n, axis=(1, 2))
        n_s_a = np.sum(self.n, axis=2)
        prev_sel_s = np.argwhere(n_s > 0).flatten()
        if prev_sel_s.size > 0:
            for k in range(self.n_planning_updates):
                random_s = np.random.choice(prev_sel_s)
                possible_actions = np.argwhere(n_s_a[random_s] > 0).flatten()
                random_a = np.random.choice(possible_actions)
                s_next_model, r_model = self.simulate_model(random_s, random_a)
                self.Q[random_s, random_a] += self.learning_rate * (r_model + self.gamma * np.max(self.Q[s_next_model])
                                                                    - self.Q[random_s, random_a])


class PrioritizedSweepingAgent(Agent):

    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon, n_planning_updates, max_queue_size=200,
                 priority_cutoff=0.01):
        super().__init__(n_states, n_actions, learning_rate, gamma, epsilon, n_planning_updates)
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue(maxsize=max_queue_size)
        self.priority = lambda s, a, r, s_next: np.abs(r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])

    def update(self, s, a, r, s_next, done):
        super().update(s, a, r, s_next, done)
        p = self.priority(s, a, r, s_next)
        if p > self.priority_cutoff:
            self.queue.put((-p, (s, a)))
        for k in range(self.n_planning_updates):
            if self.queue.empty():
                break
            _, (s_model, a_model) = self.queue.get()
            s_next_model, r_model = self.simulate_model(s_model, a_model)
            self.Q[s_model, a_model] += self.learning_rate * (r_model + self.gamma * np.max(self.Q[s_next_model])
                                                              - self.Q[s_model, a_model])
            for s_prev, a_prev in zip(*np.where(self.n[:, :, s_model] > 0)):  # not sure if this is right
                r_prev = self.r_function(s_prev, a_prev, s_model)
                p = self.priority(s_prev, a_prev, r_prev, s_model)
                if p > self.priority_cutoff:
                    self.queue.put((-p, (s_prev, a_prev)))


def test():
    n_time_steps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'ps'  # 'dyna'
    epsilon = 0.01
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma, epsilon,
                       n_planning_updates)  # Initialize Dyna policy
    elif policy == 'ps':
        pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma, epsilon,
                                      n_planning_updates, max_queue_size=0)  # Init PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = True

    for t in range(n_time_steps):
        # Select action, transition, update policy
        a = pi.select_action(s)
        s_next, r, done = env.step(a)
        pi.update(s=s, a=a, r=r, done=done, s_next=s_next)

        # Render environments
        if plot:
            env.render(Q_sa=pi.Q, plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)

        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next


if __name__ == '__main__':
    test()
