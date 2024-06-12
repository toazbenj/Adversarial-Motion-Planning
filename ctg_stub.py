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
        [[float('inf'), float('inf')], [float('inf'), float('inf')]],
        [[float('inf'), float('inf')], [float('inf'), float('inf')]],
        [[0, 1], [-1, 0]],
        [[float('inf'), float('inf')], [float('inf'), float('inf')]],
        [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    ],
    [
        [[float('inf'), float('inf')], [float('inf'), float('inf')]],
        [[-1, 0], [-2, 1]],
        [[0, 1], [-1, 0]],
        [[1, 2], [0, 1]],
        [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    ]
])

G = F
# Compute the cost-to-go values
V = generate_cost_to_go(K, G, F)

# Print the results
for k in range(K + 2):
    print("V[{}] = {}".format(k, V[k]))
