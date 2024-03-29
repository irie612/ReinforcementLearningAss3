#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from Helper import LearningCurvePlot, smooth, ComparisonPlot
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from MBRLEnvironment import WindyGridworld
import time 

def run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, learning_rate, gamma,
                    epsilon, n_planning_updates, print_time=False):
    # Write all your experiment code here
    # Look closely at the code in test() of MBRLAgents.py for an example of the execution loop
    # Log the obtained rewards during a single training run of n_timesteps, and repeat this process n_repetitions times
    # Average the learning curves over repetitions, and then additionally smooth the curve
    # Be sure to turn environment rendering off! It heavily slows down your runtime
    learning_curve = np.zeros(shape=(n_repetitions, n_timesteps))
    
    avg_time = 0
    
      
        
    for rep in range(n_repetitions):
        
        if print_time:
            initial_time = time.process_time()  
        
        env = WindyGridworld()
        if policy == 'Dyna':
            pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma, epsilon,
                           n_planning_updates)  # Initialize Dyna policy
        elif policy == 'Prioritized Sweeping':
            pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma, epsilon,
                                          n_planning_updates, max_queue_size=0)  # Init PS policy
        else:
            raise KeyError('Policy {} not implemented'.format(policy))

        s = env.reset()

        for t in range(n_timesteps):
            # Select action, transition, update policy      
            a = pi.select_action(s)
            s_next, r, done = env.step(a)
            learning_curve[rep, t] = r
            pi.update(s=s, a=a, r=r, done=done, s_next=s_next)
            if done:
                s = env.reset()
            else:
                s = s_next
        avg_time += time.process_time() - initial_time
    
    print(policy,": ",learning_rate, epsilon, n_planning_updates,"  time")
    print(avg_time/n_repetitions)
        
    learning_curve = np.mean(learning_curve, axis=0)

    # Apply additional smoothing
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    return learning_curve


def experiment():
    n_timesteps = 10000
    n_repetitions = 10
    smoothing_window = 101
    gamma = 0.99
    epsilon_experiment = True
    n_planning_updates_experiment = True
    learning_rate_experiment = True
    ql_vs_dyna = True

    for policy in ['Dyna', 'Prioritized Sweeping']:
        ##### Assignment a: effect of epsilon ######
        if epsilon_experiment:
            print(policy + ' effect of epsilon')
            learning_rate = 0.5
            n_planning_updates = 5
            epsilons = [0.01, 0.05, 0.1, 0.25]
            Plot = LearningCurvePlot(title='{}: effect of $\epsilon$-greedy'.format(policy))

            for epsilon in epsilons:
                learning_curve = run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window,
                                                 learning_rate, gamma, epsilon, n_planning_updates)
                np.savetxt(f"{policy}_{epsilon}_egreedy.csv", learning_curve, delimiter=",")
                Plot.add_curve(learning_curve, label='$\epsilon$ = {}'.format(epsilon))
            Plot.save('{}_egreedy.png'.format(policy))

        ##### Assignment b: effect of n_planning_updates ######
        if n_planning_updates_experiment:
            print(policy + ' effect of n_planning_updates')
            epsilon = 0.05
            n_planning_updatess = [1, 5, 15]
            learning_rate = 0.5
            Plot = LearningCurvePlot(title='{}: effect of number of planning updates per iteration'.format(policy))

            for n_planning_updates in n_planning_updatess:
                learning_curve = run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window,
                                                 learning_rate, gamma, epsilon, n_planning_updates)
                np.savetxt(f"{policy}_{n_planning_updates}_n_planning_updates.csv", learning_curve, delimiter=",")
                Plot.add_curve(learning_curve, label='Number of planning updates = {}'.format(n_planning_updates))
            Plot.save('{}_n_planning_updates.png'.format(policy))

        ##### Assignment c: effect of learning_rate ######
        if learning_rate_experiment:
            print(policy + ' effect of learning_rate')
            epsilon = 0.05
            n_planning_updates = 5
            learning_rates = [0.1, 0.5, 1.0]
            Plot = LearningCurvePlot(title='{}: effect of learning rate'.format(policy))

            for learning_rate in learning_rates:
                learning_curve = run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window,
                                                 learning_rate, gamma, epsilon, n_planning_updates)
                np.savetxt(f"{policy}_{learning_rate}_learning_rate.csv", learning_curve, delimiter=",")
                Plot.add_curve(learning_curve, label='Learning rate = {}'.format(learning_rate))
            Plot.save('{}_learning_rate.png'.format(policy))

    if ql_vs_dyna:
        learning_rate = 0.5
        n_planning_updatess = [0, 15]
        epsilon = 0.05
        Plot = LearningCurvePlot(title='Dyna vs Q-Learning performance'.format(policy))
        for n_planning_updates in n_planning_updatess:
                learning_curve = run_repetitions('Dyna', n_repetitions, n_timesteps, smoothing_window,
                                                 learning_rate, gamma, epsilon, n_planning_updates, True)
                if n_planning_updates == 0:
                    
                    Plot.add_curve(learning_curve, label='Q-Learning'.format(n_planning_updates))
                if n_planning_updates == 15:
                    Plot.add_curve(learning_curve, label='Dyna'.format(n_planning_updates))
        Plot.save('ql_vs_dyna.png')


def comparison(dyna_egreedy, dyna_learning_rate, dyna_n_plans, ps_egreedy, ps_learning_rate, ps_n_plans):
    n_timesteps = 10000
    n_repetitions = 10
    smoothing_window = 101
    gamma = 0.99

    epsilons = [0.01, 0.05, 0.1, 0.25]
    epsilon_dyna = epsilons[np.argmax(np.mean(dyna_egreedy, axis=1))]
    epsilon_ps = epsilons[np.argmax(np.mean(ps_egreedy, axis=1))]

    learning_rates = [0.1, 0.5, 1.0]
    learning_rate_dyna = learning_rates[np.argmax(np.mean(dyna_learning_rate, axis=1))]
    learning_rate_ps = learning_rates[np.argmax(np.mean(ps_learning_rate, axis=1))]

    n_planning_updates = [1, 5, 15]
    n_planning_updates_dyna = n_planning_updates[np.argmax(np.mean(dyna_n_plans, axis=1))]
    n_planning_updates_ps = n_planning_updates[np.argmax(np.mean(ps_n_plans, axis=1))]

    dyna_learning_curve = run_repetitions("Dyna", n_repetitions, n_timesteps, smoothing_window, learning_rate_dyna, gamma, epsilon_dyna, n_planning_updates_dyna)
    ps_learning_curve = run_repetitions("Prioritized Sweeping", n_repetitions, n_timesteps, smoothing_window, learning_rate_ps, gamma, epsilon_ps, n_planning_updates_ps)
    Plot = LearningCurvePlot(title=f"Comparison of Dyna and Prioritized sweeping with optimized parameters")
    Plot.add_curve(dyna_learning_curve, label=f"Dyna")
    Plot.add_curve(ps_learning_curve, label=f"Prioritized Sweeping")
    Plot.save("comparison_plot.png")


if __name__ == '__main__':
    experiment()
