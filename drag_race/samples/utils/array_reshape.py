"""
Array Reshape

Testing expanding array to fit given set of dimensions, repeats values as needed. Used for cost to go calculation
"""

import numpy as np

# Original array
V = np.array([1, 2, 3, 4, 5])  # Example array

# Reshape and expand into (5, 2, 2)
expanded_V = np.repeat(V[:, np.newaxis, np.newaxis], 4).reshape(5, 2, 2)

print(expanded_V)
