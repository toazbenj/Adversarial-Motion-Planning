"""
CTG Stub

Cost to go and optimal control input function testing with small, hardcoded dynamics and cost matrices
"""


import numpy as np


def generate_cost_to_go(k, cost, dynamics):
    # Initialize V with zeros
    V = np.zeros((k+2, len(dynamics[0])))

    # Iterate backwards from k to 1
    for stage in range(k, -1, -1):
        # Calculate Vminmax and Vmaxmin
        V_last = V[stage+1]
        shape_tup = cost[stage].shape
        repeat_int = np.prod(shape_tup)//np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        Vminmax = np.min(np.max(cost[stage] + V_expanded, axis=2), axis=1)
        Vmaxmin = np.max(np.min(cost[stage] + V_expanded, axis=1), axis=1)

        # Check if saddle-point can be found
        if not np.array_equal(Vminmax, Vmaxmin):
            print("Must find mixed policy")

        # Assign Vminmax to V[k-1]
        V[stage] = Vminmax

    return V


def optimal_actions(k, cost, ctg, dynamics, initial_state):
    control1 = np.zeros(k+1)
    control2 = np.zeros(k+1)
    states = np.zeros(k+2)
    states[0] = initial_state

    for stage in range(k+1):
        stage_cost = cost[stage]
        stage_dynamics = dynamics[stage]

        V_last = ctg[stage+1]
        shape_tup = stage_cost.shape
        repeat_int = np.prod(shape_tup)//np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        # gives state value but should be state number
        control1[stage] = np.argmin(np.max(stage_cost[int(states[stage])] + V_expanded[int(states[stage])], axis=1), axis=0)
        control2[stage] = np.argmax(np.min(stage_cost[int(states[stage])] + V_expanded[int(states[stage])], axis=0), axis=0)

        states[stage + 1] = stage_dynamics[int(states[stage]), int(control1[stage]), int(control2[stage])]

    return control1, control2, states


# Example usage:
K = 1
nX = 5
nU = 2
nD = 2

# Create some example G and F arrays
# G = [np.random.randint(0, 10, (nX, nU, nD)) for _ in range(K)]
# F = [np.random.randint(0, nX, (nX, nU, nD)) for _ in range(K)]

F = np.array([
    [
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[2, 3], [1, 2]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, np.nan], [np.nan, np.nan]]
    ],
    [
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[1, 2], [0, 3]],
        [[2, 3], [1, 2]],
        [[3, 4], [2, 3]],
        [[np.nan, np.nan], [np.nan, np.nan]]
    ]
])

# F = np.array([
#     [
#         [[np.nan, np.nan], [np.nan, np.nan]],
#         [[np.nan, np.nan], [np.nan, np.nan]],
#         [[0, 1], [-1, 0]],
#         [[np.nan, np.nan], [np.nan, np.nan]],
#         [[np.nan, np.nan], [np.nan, np.nan]]
#     ],
#     [
#         [[np.nan, np.nan], [np.nan, np.nan]],
#         [[-1, 0], [-2, 1]],
#         [[0, 1], [-1, 0]],
#         [[1, 2], [0, 1]],
#         [[np.nan, np.nan], [np.nan, np.nan]]
#     ]
# ])

G = np.array([
    [
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[0, 1], [-1, 0]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, np.nan], [np.nan, np.nan]]
    ],
    [
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[-1, 0], [-2, 1]],
        [[0, 1], [-1, 0]],
        [[1, 2], [0, 1]],
        [[np.nan, np.nan], [np.nan, np.nan]]
    ]
])
# Compute the cost-to-go values
V = generate_cost_to_go(K, G, F)

# Print the results
for k in range(K + 2):
    print("V[{}] = {}".format(k, V[k]))

# print('\n')
u, d, states = optimal_actions(K, G, V, F, 2)
print('u =', u)
print('d =', d)
print('x =', states)