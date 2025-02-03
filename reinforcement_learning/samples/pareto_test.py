import numpy as np
import matplotlib.pyplot as plt

# Define the payoff matrices
A1 = np.array([[3, 1, 4],
               [0, 2, 5],
               [8, 0, 3]])
B1 = np.array([[2, 0, 3],
               [1, 3, 2],
               [4, 1, 0]])

# Efficiently extract payoff pairs
flattened_payoffs = np.column_stack((A1.ravel(), B1.ravel()))

# Sort by Player A's payoff (first column)
sorted_payoffs = flattened_payoffs[np.argsort(flattened_payoffs[:, 0])]

# Efficient Pareto front computation (O(n log n))
pareto_front = []
max_b = np.inf  # Track maximum value for Player B
max_a = np.inf  # Track maximum value for Player B

pareto_index = 0
low_a = np.inf
low_b = np.inf
for point in reversed(sorted_payoffs):  # Traverse high-to-low A values
    if point[0] < low_a or point[1] < low_b:  # If not dominated in B
        pareto_index += 1

pareto_front = np.array(pareto_front[:pareto_index])

# Plot the Pareto front
plt.figure(figsize=(10, 6))
plt.scatter(flattened_payoffs[:, 0], flattened_payoffs[:, 1], alpha=0.3, label='Payoff Pairs')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front')
plt.xlabel('Payoff for Player A')
plt.ylabel('Payoff for Player B')
plt.title('Pareto Front for Mixed Strategies')
plt.legend()
plt.grid(True)
plt.show()
