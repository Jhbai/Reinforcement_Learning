def Method(transition_info: dict, nS: int, nA: int, gamma: float, epsilon: float) -> tuple:
    """
    Value Iteration algorithm for solving Markov Decision Processes (MDPs).

    Args:
        [dict] transition_info: eq. transition_info[state][action] = [(prob, next_state, reward, done)]
        [int] nS: The number of states in the MDP.
        [int] nA: The number of actions in the MDP.
        [float] gamma: The discount factor.
        [float] epsilon: The convergence threshold.

    Returns:
        [tuple] (π(s), V(s))
    """
    # Initialize value function
    V = [0] * nS # V(s) = 0 for all states
    policy = [0] * nS

    # Value Iteration
    while True:
        delta = 0
        for s in range(nS): # for all states
            V_old = V[s] # 儲存舊的value function
            action_values = [0] * nA # Q(s, a) = 0 for all actions
            for a in range(nA): # for all actions
                for prob, s_next, reward, done in transition_info[s][a]: # p(s1 | s0, a0), s1, R(s0, a0, s1), done
                    action_values[a] += prob * (reward + gamma * V[s_next]) # Q(s, a) = Σ P(s1 | s, a) * (R(s, a, s1) + γ * V(s1))
            V[s] = max(action_values) # V(s) = max_a Q(s, a)
            delta = max(delta, abs(V_old - V[s]))
        if delta < epsilon:
            break

    # Extract policy
    for s in range(nS): # for all states
        action_values = [0] * nA # Q(s, a) = 0 for all actions
        for a in range(nA): # for all actions
            for prob, s_next, reward, done in transition_info[s][a]:  # p(s1 | s0, a0), s1, R(s0, a0, s1), done
                action_values[a] += prob * (reward + gamma * V[s_next]) # Q(s, a) = Σ P(s1 | s, a) * (R(s, a, s1) + γ * V(s1))
        policy[s] = action_values.index(max(action_values)) # π(s) = argmax_a Q(s, a)

    return V, policy

            