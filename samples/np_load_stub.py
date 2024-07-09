"""https://stackoverflow.com/questions/37996295/how-to-save-numpy-array-into-computer-for-later-use-in-python"""

#Store a single array
import numpy as np
np.save('123', np.array([[1, 2, 3], [4, 5, 6]]))
print(np.load('123.npy'))

#Store multiple arrays as compressed data to disk, and load it again:
a=np.array([[1, 2, 3], [4, 5, 6]])
b=np.array([1, 2])
np.savez('123.npz', a=a, b=b)
data = np.load('123.npz')

a1 = data['a']
print(a1)
b1 = data['b']
print(b1)