"""
Visualization of theoretical values for games played with different combinations of player policy types
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import scipy_solve

# Define matrices
As = np.array([[0, 5, 0],
               [20, 3, 8],
               [20, 6, 20]])
Bs = np.array([[6, 5, 3],
               [20, 0, 8],
               [20, 0, 20]])
Ar = np.array([[1, 5, 1],
               [10, -1, 5],
               [10, -1, 10]])
Br = np.array([[-1, 5, -1],
               [10, 1, 5],
               [10, 1, 10]])

# Calculate values using scipy_solve
ys, zs, ps_star, qs_star = scipy_solve(As, Bs)
yr, zr, pr_star, qr_star = scipy_solve(Ar, Br)

ps = yr @ As @ zr
qs = yr @ Bs @ zr

pr = ys @ Ar @ zs
qr = ys @ Br @ zs

yrs = (ys + yr) / 2
zrs = (zs + zr) / 2

prs = (pr + pr_star) / 2
qrs = (qr + qr_star) / 2

psr = (ps + ps_star) / 2
qsr = (qs + qs_star) / 2

# 2 moderates
pr_mm = yrs @ Ar @ zrs
ps_mm = yrs @ As @ zrs

# moderate/conservative
pr_mc = yrs @ Ar @ zs
ps_mc = yrs @ As @ zs

# moderate/aggressive
pr_ma = yrs @ Ar @ zr
ps_ma = yrs @ As @ zr

# Prepare points for plotting
points = [
    (pr_star, ps, 'Both Aggressive Solution p1'),
    (prs, psr, 'Average p1'),
    (pr, ps_star, 'Both Conservative Solution p1'),
    (pr_mm, ps_mm, 'Both Moderate p1'),
    (pr_mc, ps_mc, "Moderate/Conservative p1"),
    (pr_ma, ps_ma, "Moderate/Aggressive p1"),
]

# Plot the results
def plot_pareto_front(points):
    plt.figure(figsize=(10, 6))
    for point in points:
        plt.scatter(point[0], point[1], label=point[2])
    plt.title('Pareto Front')
    plt.xlabel('Rank (r)')
    plt.ylabel('Safety (s)')
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

plot_pareto_front(points)
print("player 1")
[print(pt) for pt in points]