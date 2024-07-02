"""
Potential Game Stub

Functions for solving bimatrix games and finding mixed policies (not currently working)
"""
import numpy as np
from scipy.optimize import linprog, minimize

def scipy_solve(A, B):
    num_strategies_A = A.shape[0]
    num_strategies_B = A.shape[1]

    # Objective function for Player 1
    c1 = -A.sum(axis=1)

    # Constraints for Player 1
    A_ub1 = np.ones((1, num_strategies_A))
    b_ub1 = [1]

    # Bounds for the probabilities (between 0 and 1)
    bounds1 = [(0, 1)] * num_strategies_A

    # Solve the LP for Player 1
    res1 = linprog(c1, A_ub=A_ub1, b_ub=b_ub1, bounds=bounds1, method='highs')

    # Objective function for Player 2
    c2 = -B.sum(axis=0)

    # Constraints for Player 2
    A_ub2 = np.ones((1, num_strategies_B))
    b_ub2 = [1]

    # Bounds for the probabilities (between 0 and 1)
    bounds2 = [(0, 1)] * num_strategies_B

    # Solve the LP for Player 2
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')

    # Get the mixed strategies
    player1_strategy = res1.x
    player2_strategy = res2.x

    return player1_strategy, player2_strategy, (res1.x[-1], res2.x[-1])


def scipy_solve_quadratic(A, B):
    m, n = A.shape
    x0 = np.random.rand(n + m + 2)

    # Construct H matrix
    H_top = np.hstack((np.zeros((m, m)), A + B, np.zeros((m, 2))))
    H_middle = np.hstack((A.T + B.T, np.zeros((n, n + 2))))
    H_bottom = np.zeros((2, m + n + 2))
    H = np.vstack((H_top, H_middle, H_bottom))

    # Construct c vector
    c = np.hstack((np.zeros(m + n), -1, -1))

    # Inequality constraints: Ain * x >= bin
    Ain_top = np.hstack((np.zeros((m, m)), -A, np.ones((m, 1)), np.zeros((m, 1))))
    Ain_bottom = np.hstack((-B.T, np.zeros((n, n + 1)), np.ones((n, 1))))
    Ain = np.vstack((Ain_top, Ain_bottom))
    bin = np.zeros(m + n)

    # Equality constraints: Aeq * x = beq
    Aeq = np.vstack((np.hstack((np.ones(m), np.zeros(n + 2))), np.hstack((np.zeros(m), np.ones(n), 0, 0))))
    beq = np.array([1, 1])

    # Bounds for the variables
    low = np.hstack((np.zeros(n + m), -np.inf, -np.inf))
    high = np.hstack((np.ones(n + m), np.inf, np.inf))

    bounds = [(low[i], high[i]) for i in range(len(low))]

    # Objective function for minimize (since quadprog minimizes 0.5 * x^T H x + c^T x)
    def objective(x):
        # check c, transpose?
        return 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(c, x)

    # Constraints for minimization
    cons = ({'type': 'ineq', 'fun': lambda x: np.dot(Ain, x) - bin},
            {'type': 'eq', 'fun': lambda x: np.dot(Aeq, x) - beq})

    # Solve the quadratic program
    # check options, match matlab?
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-8, 'eps': 1e-8})

    x = result.x
    y = x[:m]
    z = x[m:m + n]
    p = x[m + n]
    q = x[m + n + 1]

    return y, z, p, q


def scipy_solve2(A, B):
    m, n = A.shape

    # Construct the objective function vector
    c = np.zeros(m + n + 2)
    c[-2:] = -1

    # Construct the inequality constraint matrix Ain and vector bin
    Ain_top = np.hstack((np.zeros((m, m)), -A, np.ones((m, 1)), np.zeros((m, 1))))
    Ain_bottom = np.hstack((-B.T, np.zeros((n, n + 1)), np.ones((n, 1))))
    Ain = np.vstack((Ain_top, Ain_bottom))
    bin = np.zeros(m + n)

    # Construct the equality constraint matrix Aeq and vector beq
    Aeq = np.zeros((2, m + n + 2))
    Aeq[0, :m] = 1
    Aeq[1, m:m + n] = 1
    beq = np.array([1, 1])

    # Define the bounds for the variables
    bounds = [(0, 1)] * (m + n) + [(-np.inf, np.inf), (-np.inf, np.inf)]

    # Solve the linear program
    result = linprog(c, A_ub=Ain, b_ub=bin, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')

    if result.success:
        x = result.x
        y = x[:m]
        z = x[m:m + n]
        p = x[m + n]
        q = x[m + n + 1]
        return y, z, p, q
    else:
        raise ValueError("Linear programming did not converge")


# Example usage
# A = np.array([[2, 30],
#               [0, 8]])
# B = np.array([[2, 0],
#               [30, 8]])
# A = np.array([[-2, 1],
#               [0, -1]])
# B = np.array([[-1, 2],
#               [3, -2]])

A = np.array([[2, 0, 1],
              [2, -1, 0],
              [2, -2, 2]])
B = np.array([[-2, 0, -1],
              [2, 1, 0],
              [2, 2, 2]])

# player1_strategy, player2_strategy, value = scipy_solve(A, B)
player1_strategy, player2_strategy, value1, value2 = scipy_solve2(A, B)

print(f"Player 1 strategy: \n{player1_strategy}")
print(f"Player 2 strategy: \n{player2_strategy}")
print("Game Value1: ", value1)
print("Game Value2: ", value2)
# print("Game Value: ", value)

# s1_star, s2_star, value = solve_bimatrix_potential_game(A, B)
#
# if s1_star is not None:
#     print("Player 1's mixed strategy:", s1_star)
#     print("Player 2's mixed strategy:", s2_star)
#     print("Value of the game:", value)
# else:
#     print("No Nash equilibrium found.")