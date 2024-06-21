"""
Tensor Multiplication

Testing numpy matrix multiplication for 3+ dimensions
"""

import numpy as np
import scipy

# Generate random matrices
P = np.random.rand(144, 9, 9)
T = np.random.rand(144, 2, 2)

result = np.tensordot(P, T, axes=([2], [1]))

# Print the shapes of the resulting matrices
print("Shape of result after tensordot:", result.shape)
print(result)

