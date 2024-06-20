import numpy as np

def optimal_actions(k, cost, ctg, dynamics, initial_state):
    control1 = np.zeros(k + 1, dtype=int)
    control2 = np.zeros(k + 1, dtype=int)
    states_played = np.zeros(k + 2, dtype=int)
    states_played[0] = initial_state

    for stage in range(k + 1):
        V_last = ctg[stage + 1]
        shape_tup = cost.shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        # Correct control1 and control2 calculation
        control1[stage] = np.argmin(np.nanmax(cost[states_played[stage], :, :] + V_expanded[states_played[stage], :, :], axis=1))
        control2[stage] = np.argmax(np.nanmin(cost[states_played[stage], :, :] + V_expanded[states_played[stage], :, :], axis=0))

        states_played[stage + 1] = dynamics[states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states_played

# Example usage
k = 3
cost = np.random.rand(5, 5, 5)
ctg = np.random.rand(k + 2, 5)
dynamics = np.random.randint(0, 5, (5, 5, 5))
initial_state = 2

control1, control2, states_played = optimal_actions(k, cost, ctg, dynamics, initial_state)
print(f"Control 1: {control1}")
print(f"Control 2: {control2}")
print(f"States Played: {states_played}")
