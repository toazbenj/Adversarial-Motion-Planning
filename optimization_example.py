'''
Testing out scipy optimization for mixed policy calculations

https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.optimize.linprog.html
'''

from scipy.optimize import linprog

c = [-1, 4]
A = [[-3, 1],
     [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')

print(res.fun, '\n')
print(res.x, '\n')
print(res.message)