import numpy as np

def reward_function(s, env_size):
    goal_state = np.array([env_size - 1, env_size - 1], dtype=np.uint8)
    return 1.0 if np.array_equal(s, goal_state) else 0.0

def transition_probabilities(env, s, a, env_size, directions, holes):
    cells = [s + directions[a], s + directions[(a-1) % 4]]
    probs = [1/2, 1/2]
    
    prob_next_state = np.zeros((env_size, env_size))

    for i in range(len(cells)):
        cell = cells[i]
        prob = probs[i]
        if 0 <= cell[0] < env_size and 0 <= cell[1] < env_size:
            prob_next_state[cell[0], cell[1]] = prob

    return prob_next_state
