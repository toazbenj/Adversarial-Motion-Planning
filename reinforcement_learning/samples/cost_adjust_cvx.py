import numpy as np
import cvxpy as cp


def cost_adjustment(A, B, global_min_position):
    """Adjusts the cost matrices to ensure potential function constraints"""

    # Compute initial potential function
    phi_initial = global_potential_function_numeric(A, B, global_min_position)

    if is_valid_exact_potential(A, B, phi_initial, global_min_position) and \
            is_global_min_enforced(phi_initial, global_min_position):
        E = np.zeros_like(A)
        return E

    # Convex optimization to find Ea
    m, n = A.shape
    E = cp.Variable((m, n))
    phi = cp.Variable((m, n))
    A_prime = A + E
    constraints = []

    # Constraint 1: Ensure global minimum position is zero
    constraints.append(phi[global_min_position[0], global_min_position[1]] == 0)

    # Constraint 2: Enforce non-negativity
    epsilon = 1e-6
    for k in range(m):
        for j in range(n):
            if (k, j) != tuple(global_min_position):
                constraints.append(phi[k, j] >= epsilon)

    # Constraint 3: Ensure exact potential condition
    for k in range(1, m):
        for l in range(n):
            delta_A = A_prime[k - 1, l] - A_prime[k, l]
            delta_phi = phi[k - 1, l] - phi[k, l]
            constraints.append(delta_A == delta_phi)

    for k in range(m):
        for l in range(1, n):
            delta_B = B[k, l] - B[k, l - 1]
            delta_phi = phi[k, l] - phi[k, l - 1]
            constraints.append(delta_B == delta_phi)

    # Solve optimization problem
    objective = cp.Minimize(cp.norm(E, 'fro'))
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.SCS, max_iters=50000, eps=1e-6, verbose=False)
    # problem.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=10000)
    # problem.solve(solver=cp.GUROBI, verbose=False)
    # problem.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-9})

    return E.value


def global_potential_function_numeric(A, B, global_min_position):
    """Computes a global potential function for given cost matrices"""
    m, n = A.shape
    phi = np.zeros((m, n))

    for i in range(1, m):
        phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]

    for j in range(1, n):
        phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]

    for i in range(1, m):
        for j in range(1, n):
            phi[i, j] = (phi[i - 1, j] + A[i, j] - A[i - 1, j] +
                         phi[i, j - 1] + B[i, j] - B[i, j - 1]) / 2

    return phi - phi[global_min_position[0], global_min_position[1]]


def is_valid_exact_potential(A, B, phi, global_min_position):
    """Checks if the exact potential condition is met"""
    m, n = A.shape

    for i in range(1, m):
        for j in range(n):
            if abs((A[i, j] - A[i - 1, j]) - (phi[i, j] - phi[i - 1, j])) > 1e-6:
                return False

    for i in range(m):
        for j in range(1, n):
            if abs((B[i, j] - B[i, j - 1]) - (phi[i, j] - phi[i, j - 1])) > 1e-6:
                return False

    return True


def is_global_min_enforced(phi, global_min_position):
    """Checks if the global minimum is enforced"""
    m, n = phi.shape
    if phi[global_min_position[0], global_min_position[1]] != 0:
        return False

    for i in range(m):
        for j in range(n):
            if (i, j) != tuple(global_min_position) and phi[i, j] <= 0:
                return False

    return True


# ======= Test Matrices =======


# A1 = np.array([[0, 1, 2],
#                [-1, 0, 1],
#                [-2, -1, 0]])
# A2 = np.array([[0, 1, 2],
#                [1, 2, 3],
#                [2, 3, 4]])
#
# B1 = np.array([[2, 1, 0],
#                [2, 1, 0],
#                [2, 1, 0]])
# B2 = np.array([[0, 0, 0],
#                [0, 0, 0],
#                [0, 0, 1]])
# size = 30
# A1 = np.random.uniform(0, 50, (size, size))
# B1 = np.random.uniform(0, 50, (size, size))

A1 = np.load(('(0, 255, 0)scalar.npz'))['arr']
B1 = np.load(('(0, 0, 255)scalar.npz'))['arr'].transpose()

# Compute potential functions for adjusted costs
potential_functions = []

# for i in range(len(player1_games)):
#     for k in range(A1.shape[0]):
#         for j in range(A1.shape[1]):
#             global_min_position = [k, j]
#             player1_errors, _ = cost_adjustment(player1_games, player2_games, global_min_position)
#
#             if player1_errors[i] is None:
#                 continue
#
#             # Adjusted cost matrix for Player 1
#             player1_adjusted = player1_games[i] + player1_errors[i]
#
#             # Compute potential function for adjusted costs
#             phi = global_potential_function_numeric(player1_adjusted, player2_games[i], global_min_position)
#             potential_functions.append(phi)
#
#             # Validation check
#             if is_global_min_enforced(phi, global_min_position) and phi[
#                 global_min_position[0], global_min_position[1]] == 0:
#                 print(f"Subgame {i + 1} Results:")
#                 print("Player 1 A_prime:")
#                 print(player1_adjusted)
#                 print("Potential Function:")
#                 print(phi)
#                 print("Global Min:", global_min_position)
#                 print("Global Minimum Enforced:", is_global_min_enforced(phi, global_min_position))
#                 print("Exact Potential:", phi[global_min_position[0], global_min_position[1]] == 0)
#                 print("----------------------------------------")
# print("==========================================================================")

player2_sec = np.argmin(np.max(B1, axis = 0), axis=0)
for k in range(A1.shape[0]):
    global_min_position = [k, player2_sec]
    E = cost_adjustment(A1, B1, global_min_position)

    print(k)

    if E is None:
        continue

    # Adjusted cost matrix for Player 1
    A_prime = A1 + E

    # Compute potential function for adjusted costs
    phi = global_potential_function_numeric(A_prime, B1, global_min_position)
    potential_functions.append(phi)

    # Validation check
    if is_global_min_enforced(phi, global_min_position) and phi[
        global_min_position[0], global_min_position[1]] == 0:
        print(f"Subgame {1} Results:")
        print("Player 1 A_prime:")
        print(A_prime)
        print("Potential Function:")
        print(phi)
        print("Global Min:", global_min_position)
        print("Global Minimum Enforced:", is_global_min_enforced(phi, global_min_position))
        print("Exact Potential:", phi[global_min_position[0], global_min_position[1]] == 0)
        print("----------------------------------------")
print("==========================================================================")