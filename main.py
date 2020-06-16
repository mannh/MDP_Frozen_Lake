# -*- coding: utf-8 -*-

from World import World
import numpy as np
import pandas as pd
from itertools import product


world = World()
n_actions = 4  # Assuming grid world there are only 4 possible actions.
n_states = world.get_nstates()
terminal_states = world.get_stateHoles() + world.get_stateGoal()
n_rows = world.get_ncols()
n_cols = world.get_nrows()
p = .8
all_actions = [1, 2, 3, 4]


def create_state_rewards(step):
    return np.array([1 if i in world.get_stateGoal() else -1 if i in world.get_stateHoles() else step for i in
                     range(1, n_states + 1)])


def create_state_transition_matrices():
    states = range(1, n_states+1)
    # each action has a matrix
    transition_matrices_dict = {i: pd.DataFrame(np.zeros((n_states, n_states)), index=states,
                                                columns=states) for i in range(1, n_actions+1)}
    for i, j in product(range(n_rows), range(n_cols)):
        _from = map_location_to_state(i, j)
        if _from not in terminal_states:
            for action in all_actions:
                for _a in [action-1, action, action+1]:  # for each action it is possible to go left or right
                    _a = get_action(_a)
                    _p = p if _a == action else (1-p)/2
                    dest_i, dest_j = move(i, j, _a)
                    if is_legal_index(dest_i, dest_j):
                        to = map_location_to_state(dest_i, dest_j)
                        transition_matrices_dict[action].loc[_from, to] += _p
                    else:  # reflective boundaries
                        transition_matrices_dict[action].loc[_from, _from] += _p
        else:
            for action in all_actions:
                transition_matrices_dict[action].loc[_from, _from] += 1  # Terminal state is an absorbing state


    return transition_matrices_dict


def is_legal_index(i, j):
    return i in range(n_rows) and j in range(n_cols)


def map_location_to_state(i, j):
    return n_rows * j + i + 1


def move(i, j, a):
    return {
        1: (i-1, j),  # N
        2: (i, j+1),  # E
        3: (i+1, j),  # S
        4: (i, j-1)   # W
    }[a]


def get_action(a):
    actions = [1, 2, 3, 4]
    #         [N, E, S, W]
    try:
        return actions[a - 1]
    except IndexError:
        return 1  # allow loop from west to north


def value_iteration(theta, gamma):
    # Initialize V (s) arbitrarily for all s ∈ S+ (e.g., V (s) = 0 for all S+)
    V = np.zeros(world.get_nstates())
    Pi = np.zeros(world.get_nstates())
    while True:
        delta = 0
        for s in range(1, world.get_nstates()+1):
            v = V[s-1].copy()
            Pi[s-1], V[s-1] = max_action_func(s, gamma, V)
            delta = max(delta, np.abs(v-V[s-1]))
        if delta < theta:
            break
    return Pi, V


def value_calc_func(s, gamma, V):
    global rewards
    if s in terminal_states:
        actions_dict = {a: 0 for a in all_actions}
    else:
        actions_dict = {a: np.dot(rewards, transition_matrices_dict[a].loc[s, :]) + gamma * np.dot(
            transition_matrices_dict[a].loc[s, :], V)
                        for a in all_actions}
    return actions_dict


def max_action_func(s, gamma, V):
    actions_dict = value_calc_func(s, gamma, V)
    max_value = -np.inf
    for a, v in actions_dict.items():
        if v > max_value:
            max_value = v
            chosen_policy = a
    return chosen_policy, max_value


def policy_iteration(theta, gamma):
    Pi = np.full((n_states, n_actions), 1/n_actions)
    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(Pi, gamma, theta)
        Pi_prime = policy_improvement(V, gamma)
        world.plot_value(V)
        world.plot_policy(np.argmax(Pi_prime, axis=1) + 1)
        if (Pi_prime == Pi).all():
            policy_stable = True
        Pi = Pi_prime
    Pi = np.argmax(Pi, axis=1) + 1
    return Pi, V


def policy_evaluation(Pi, gamma, theta):
    global rewards
    # Initialize V(s) arbitrarily for all s ∈ S+ (e.g., V (s) = 0 for all S+)
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(1, n_states+1):
            v = V[s-1].copy()
            temp_sum = 0
            rewards_sum = 0
            if s not in terminal_states:
                for a in transition_matrices_dict.keys():
                    temp_sum += Pi[s - 1, a-1] * np.dot(transition_matrices_dict[a].loc[s, :], V)
                    rewards_sum += Pi[s - 1, a-1] * np.dot(rewards, transition_matrices_dict[a].loc[s, :])
            V[s-1] = rewards_sum + gamma * temp_sum
            delta = max(delta, np.abs(V[s-1] - v))
        if delta < theta:
            return V


def policy_improvement(V, gamma):
    Q = np.zeros((n_states, n_actions))
    Pi_prime = np.zeros((n_states, n_actions))
    for s in range(1, n_states+1):
        for a in transition_matrices_dict.keys():
            if s not in terminal_states:
                temp = np.dot(transition_matrices_dict[a].loc[s, :].values, V)
            else:
                temp = 0
            Q[s-1, a-1] = np.dot(rewards, transition_matrices_dict[a].loc[s, :]) + gamma * temp
    for s in range(n_states):
        max_val = np.max(Q[s, :])
        best_actions = Q[s, :] == max_val
        Pi_prime[s, :] = best_actions/sum(best_actions)
    return Pi_prime




if __name__ == "__main__":
    transition_matrices_dict = create_state_transition_matrices()
    # section b
    rewards = create_state_rewards(step=-.04)
    Pi, V = value_iteration(theta=10**-4, gamma=1)
    world.plot_value(V)
    world.plot_policy(Pi)
    # section c
    rewards = create_state_rewards(step=-.04)
    Pi, V = value_iteration(theta=10**-4, gamma=.9)
    world.plot_value(V)
    world.plot_policy(Pi)
    # section d
    rewards = create_state_rewards(step=-.02)
    Pi, V = value_iteration(theta=10**-4, gamma=1)
    world.plot_value(V)
    world.plot_policy(Pi)
    # section e
    rewards = create_state_rewards(step=-.04)
    Pi, V = policy_iteration(theta=10**-4, gamma=.9)
    # plots are generated during the run of policy_iteration




