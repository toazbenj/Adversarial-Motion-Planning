import numpy as np


def generate_cost_to_go(k, cost, dynamics):
    # Initialize V with zeros
    V = np.zeros((k+2, len(dynamics)))

    # Iterate backwards from k to 1
    for stage in range(k, 0, -1):
        # for state_number in range(len(dynamics[0])):
        # Calculate Vminmax and Vmaxmin

        shape = (1, cost.shape[1], cost.shape[2])
        V_expanded = np.tile(V[stage], shape)
        Vminmax = np.min(np.max(cost[stage] + V_expanded, axis=2), axis=1)
        Vmaxmin = np.max(np.min(cost[stage] + V_expanded, axis=1), axis=2)

        # Vminmax = np.min(np.max(cost[stage], axis=2), axis=1) + V[stage]
        # Vmaxmin =  np.max(np.min(cost[stage], axis=1), axis=2) + V[stage]

        # Check if saddle-point can be found
        if not np.array_equal(Vminmax, Vmaxmin):
            print("Must find mixed policy")

        # Assign Vminmax to V[k-1]
        V[k - 1] = Vminmax

    # If needed, convert V to desired format or print the result
    print("Round ", k, ": ", V)

def compute_cost_to_go_alt(K, G, F):
    # Initialize the cost-to-go values
    nX = G[0].shape[0]
    V = [float('inf')] * (K + 1)
    V[K] = np.zeros((nX, 1), dtype=np.int8)

    for k in range(K-1, -1, -1):
        # Get the relevant G and F tensors
        Gk = G[k]
        Fk = F[k]

        # Broadcast Vk_plus_1 to match the dimensions of Fk
        Vk_plus_1 = V[k + 1].reshape(-1, 1, 1)

        # Compute Vk_plus_1_Fk by indexing Vk_plus_1 using Fk
        Vk_plus_1_Fk = np.take_along_axis(Vk_plus_1, Fk, axis=0)

        # Compute the cost to go for each state-action pair
        cost_to_go = Gk + Vk_plus_1_Fk

        # Find the min-max and max-min values
        Vminmax = np.min(np.max(cost_to_go, axis=2), axis=1)
        Vmaxmin = np.max(np.min(cost_to_go, axis=1), axis=1)

        # Ensure a saddle-point can be found
        if np.any(Vminmax != Vmaxmin):
            raise ValueError('Saddle-point cannot be found')

        # Set the cost-to-go value for stage k
        V[k] = Vminmax.reshape(-1, 1)

    return V

# Example usage:
K = 1
nX = 5
nU = 2
nD = 2

# Create some example G and F arrays
# G = [np.random.randint(0, 10, (nX, nU, nD)) for _ in range(K)]
# F = [np.random.randint(0, nX, (nX, nU, nD)) for _ in range(K)]
F, G = np.array([[[[float('inf'),float('inf')],
                  [float('inf'),float('inf')]],
                 [[float('inf'),float('inf')],
                  [float('inf'),float('inf')]],
                 [[0,-1],
                  [1,0]],
                 [[float('inf'),float('inf')],
                  [float('inf'),float('inf')]],
                 [[float('inf'),float('inf')],
                  [float('inf'),float('inf')]]],
                     [[[float('inf'), float('inf')],
                       [float('inf'), float('inf')]],
                      [[0, -1],
                       [1, 0]],
                      [[0, -1],
                       [1, 0]],
                      [[0, -1],
                       [1, 0]],
                      [[float('inf'), float('inf')],
                       [float('inf'), float('inf')]]]])

# Compute the cost-to-go values
V = generate_cost_to_go(K, G, F)

# Print the results
for k in range(K + 1):
    print(f"V[{k}] =\n{V[k]}")
