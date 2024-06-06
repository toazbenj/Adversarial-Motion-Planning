# https://nashpy.readthedocs.io/en/stable/how-to/use-replicator-dynamics.html

import nashpy as nash
import numpy as np

A = np.array([[0, -1, 1],
              [1, 0, -1],
              [-1, 1, 0]])

rps = nash.Game(A)
print(rps)

eqs = rps.support_enumeration()
print("\n", "NE:", list(eqs))
