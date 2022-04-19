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
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s):
        p = np.random.uniform()
        if p <= 1 - self.epsilon:
            choices = np.argwhere(s == np.max(s)).flatten()
            a = np.random.choice(choices)
        else:
            choices = np.argwhere(s != np.max(s)).flatten()
            if choices.size > 0:
                a = np.random.choice(choices)
            else:
                a = np.random.randint(s.size)
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # i think this is all right but double check
        self.n[s, a, s_next] += 1
        self.R_sum[s, a, s_next] += r
    
    def simulate_model(self, s, a):
        p_hat = self.n[s, a] / np.sum(self.n, axis=2)
        r_hat = self.R_sum[s, a] / self.n[s, a]
        s_next = np.random.choice([a in range(self.n_actions)], p=p_hat)
        r = r_hat[s_next]
        return s_next, r
        


class DynaAgent(Agent):

    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon):
        super().__init__(n_states, n_actions, learning_rate, gamma, epsilon)

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # TO DO: Add own code
        super().update(s, a, r, done, s_next, n_planning_updates)
        


class PrioritizedSweepingAgent(Agent):

    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon, max_queue_size=200, priority_cutoff=0.01):
        super().__init__(n_states, n_actions, learning_rate, gamma, epsilon)
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue(maxsize=max_queue_size)

    def update(self, s, a, r, done, s_next, n_planning_updates):
        super().update(s, a, r, done, s_next, n_planning_updates)
        # need to change np.max() to take some args. I don't understand the algorithms notationnn
        priority = r + self.gamma * np.max() - self.Q[s, a]
        if priority > 0:
            self.queue.put((-priority, (s, a)))
        for k in range(n_planning_updates):
            _, (s, a) = self.queue.get()
        # simulate model
        # update Q table

        pass


def test():
    n_time_steps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'dyna'  # 'ps'
    epsilon = 0.1
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma, epsilon)  # Initialize Dyna policy
    elif policy == 'ps':
        pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma, epsilon)  # Init PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = False

    for t in range(n_time_steps):
        # Select action, transition, update policy
        a = pi.select_action(s)
        s_next, r, done = env.step(a)
        pi.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

        # Render environment
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
