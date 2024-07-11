"""
Numpy Compare

Prototype for array_find function, testing how to find indices for value in a numpy array when value is also numpy array.
Doesn't work. Working version uses numpy.equal
"""

import numpy as np

# Example data
state_lst = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
next_state_mat = np.array([[4, 5, 6], [1, 2, 3], [10, 11, 12]])

# Convert state_lst to a numpy array for efficient comparisons
state_lst_np = np.array(state_lst)

# Find matches using np.where and np.all
matches = [np.where(np.all(next_state_mat == state, axis=1))[0] for state in state_lst_np]

# Initialize a lookup matrix with default value (e.g., -1 to indicate no match)
dynamics_lookup_mat = np.full((1, 1, 1, next_state_mat.shape[0]), -1, dtype=int)

# Fill the lookup matrix with indices of matching rows
for idx, match in enumerate(matches):
    if match.size > 0:
        for m in match:
            dynamics_lookup_mat[0, 0, 0, m] = idx

print("Dynamics Lookup Matrix:")
print(dynamics_lookup_mat)
