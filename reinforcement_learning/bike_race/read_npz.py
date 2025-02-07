import numpy as np

# Load the npz file
loaded_data = np.load('(0, 0, 255)scalar.npz')

# Extract the array
loaded_arr = loaded_data['arr']

print(loaded_arr)  # Output: [1 2 3 4 5]
