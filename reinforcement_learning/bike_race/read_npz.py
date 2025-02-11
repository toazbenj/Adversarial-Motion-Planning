import numpy as np

# Load the npz file
A1_load = np.load('A1.npz')
A2_load = np.load('A1.npz')
B_load = np.load('A1.npz')

# Extract the array
A1 = A1_load['arr']
A2 = A2_load['arr']
B = B_load['arr']

print("A1", A1)
print("A2", A2)
print("B", B)