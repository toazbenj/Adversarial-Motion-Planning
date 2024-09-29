import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Set up the grid for gamma and sigma
gamma = np.arange(0.0, 1.1, 0.1)  # Range from 0.0 to 1.0 with step 0.1
sigma = np.arange(0.0, 1.1, 0.1)  # Same for sigma

# Create meshgrid
gamma_grid, sigma_grid = np.meshgrid(gamma, sigma)

# Step 2: Calculate the payoff function
payoff = 4 * gamma_grid * sigma_grid - 2 * sigma_grid - gamma_grid + 3

# Step 3: Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(gamma_grid, sigma_grid, payoff, cmap='viridis')

# Step 4: Label the axes
ax.set_xlabel('Gamma')
ax.set_ylabel('Sigma')
ax.set_zlabel('Payoff')

# Show the plot
plt.show()
