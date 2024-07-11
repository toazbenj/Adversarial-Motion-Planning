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
pr_actual = yrs @ Ar @ zrs
ps_actual = yrs @ As @ zrs

qr_actual = yrs @ Br @ zrs
qs_actual = yrs @ Bs @ zrs

# moderate/conservative
pr_mc = yrs @ Ar @ zs
ps_mc = yrs @ As @ zs

qr_mc = yrs @ Br @ zs
qs_mc = yrs @ Bs @ zs

# moderate/aggressive
pr_ma = yrs @ Ar @ zr
ps_ma = yrs @ As @ zr

qr_ma = yrs @ Br @ zr
qs_ma = yrs @ Bs @ zr

# conservative/aggressive
pr_ca = ys @ Ar @ zr
ps_ca = ys @ As @ zr

qr_ca = ys @ Br @ zr
qs_ca = ys @ Bs @ zr

# Prepare points for plotting
points1 = [
    (pr_star, ps, 'Both Aggressive Solution p1'),
    (prs, psr, 'Average p1'),
    (pr, ps_star, 'Both Conservative Solution p1'),
    (pr_actual, ps_actual, 'Both Moderate p1'),
    (pr_mc, ps_mc, "Moderate/Conservative p1"),
    (pr_ma, ps_ma, "Moderate/Aggressive p1"),
    (pr_ca, ps_ca, "Conservative/Aggressive p1")
]

points2 = [
    (qr_star, qs, 'Both Aggressive Solution p2'),
    (qrs, qsr, 'Average p2'),
    (qr, qs_star, 'Both Conservative Solution p2'),
    (qr_actual, qs_actual, 'Both Moderate p2'),
    (qr_mc, qs_mc, "Moderate/Conservative p2"),
    (qr_ma, qs_ma, "Moderate/Aggressive p2"),
    (qr_ca, qs_ca, "Conservative/Aggressive p2")
]

# Plot the results
def plot_pareto_front(points1, points2):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(left=0.3, hspace=0.5)

    # First subplot
    for point in points1:
        axs[0].scatter(point[0], point[1], label=point[2])
    axs[0].set_title('Pareto Front - Set 1')
    axs[0].set_xlabel('Rank (r)')
    axs[0].set_ylabel('Safety (s)')
    axs[0].legend()
    axs[0].set_xlim(left=0)
    axs[0].set_ylim(bottom=0)

    # Second subplot
    for point in points2:
        axs[1].scatter(point[0], point[1], label=point[2])
    axs[1].set_title('Pareto Front - Set 2')
    axs[1].set_xlabel('Rank (r)')
    axs[1].set_ylabel('Safety (s)')
    axs[1].legend()
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

plot_pareto_front(points1, points2)
print("player 1")
[print(pt) for pt in points1]
print("player 2")
[print(pt) for pt in points2]