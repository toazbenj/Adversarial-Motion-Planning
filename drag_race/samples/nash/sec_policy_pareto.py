import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate sample data
np.random.seed(0)
n_points = 100
cost = np.random.rand(n_points)
risk = np.random.rand(n_points)
performance = np.random.rand(n_points)

# Identify Pareto front
def pareto_front(cost, risk, performance):
    is_pareto = np.ones(cost.shape[0], dtype=bool)
    for i in range(cost.shape[0]):
        is_pareto[i] = np.all(np.any([(cost[:i] > cost[i]),
                                      (risk[:i] > risk[i]),
                                      (performance[:i] > performance[i])], axis=0))
    return is_pareto

is_pareto = pareto_front(cost, risk, performance)

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cost, risk, performance, c='b', label='Policies')
ax.scatter(cost[is_pareto], risk[is_pareto], performance[is_pareto], c='r', label='Pareto Front')

ax.set_xlabel('Cost')
ax.set_ylabel('Risk')
ax.set_zlabel('Performance')
ax.set_title('Pareto Front for Security Policies')
ax.legend()

plt.show()
