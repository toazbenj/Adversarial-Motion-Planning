import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the payoff matrices
A1 = np.array([[3, 1],
               [0, 2]])
B1 = np.array([[2, 0],
               [1, 3]])
A2 = np.array([[4, 1],
               [3, 2]])
B2 = np.array([[1, 4],
               [2, 3]])

# Define the expected payoff functions
def expected_payoff(p, q, A, B):
    E_A = p*q*A[0, 0] + p*(1-q)*A[0, 1] + (1-p)*q*A[1, 0] + (1-p)*(1-q)*A[1, 1]
    E_B = p*q*B[0, 0] + p*(1-q)*B[0, 1] + (1-p)*q*B[1, 0] + (1-p)*(1-q)*B[1, 1]
    return E_A, E_B

# Generate a grid of probabilities
P = np.linspace(0, 1, 50)
Q = np.linspace(0, 1, 50)

# Calculate the payoff pairs for the mixed strategies
payoff_pairs = []
for p in P:
    for q in Q:
        E_A1, E_B1 = expected_payoff(p, q, A1, B1)
        E_A2, E_B2 = expected_payoff(p, q, A2, B2)
        payoff_pairs.append(((E_A1, E_B1), (E_A2, E_B2)))

# Flatten the array and create pairs
flattened_payoffs = np.array([list(payoff_pairs[i][j]) for i in range(len(payoff_pairs)) for j in range(2)])

# Find Pareto front
pareto_front = []
for i in range(len(flattened_payoffs)):
    dominated = False
    for j in range(len(flattened_payoffs)):
        if all(flattened_payoffs[j] <= flattened_payoffs[i]) and any(flattened_payoffs[j] < flattened_payoffs[i]):
            dominated = True
            break
    if not dominated:
        pareto_front.append(flattened_payoffs[i])

pareto_front = np.array(pareto_front)

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
