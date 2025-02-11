import numpy as np
import cvxpy as cp


def cost_adjustment(A, B, global_min_position):
    """Adjusts the cost matrices to ensure potential function constraints"""

    # Compute initial potential function
    phi_initial = potential_function(A, B, global_min_position)

    if is_valid_exact_potential(A, B, phi_initial) and \
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
            delta_A = A_prime[k , l] - A_prime[k-1, l]
            delta_phi = phi[k, l] - phi[k-1, l]
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

    # have to install things for these solvers
    # problem.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=10000)
    # problem.solve(solver=cp.GUROBI, verbose=False)
    # problem.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-9})

    return E.value


def potential_function(A, B, global_min_position):
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


def is_valid_exact_potential(A, B, phi):
    """Checks if the exact potential condition is met"""
    m, n = A.shape
    epsilon = 1e-6
    # something is wrong here:

    for i in range(1, m):
        for j in range(n):
            if not np.isclose((A[i, j] - A[i - 1, j]), (phi[i, j] - phi[i - 1, j]), atol=epsilon):
                return False

    for i in range(m):
        for j in range(1, n):
            if not np.isclose((B[i, j] - B[i, j-1]), (phi[i, j] - phi[i, j-1]), atol=epsilon):
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


def find_adjusted_costs(A1, A2, B):
    player2_sec_policy = np.argmin(np.max(B, axis=0), axis=0)

    # find worst case safety actions and avoid
    max_value = np.max(A2)
    safe_row_indices = np.where(~np.any(A2 == max_value, axis=1))[0]

    # find error matrices to make each combination of indices the global min of potential function
    E_star = np.ones_like(A1) * np.inf
    for i in safe_row_indices:
        min_position = (i, player2_sec_policy)
        E = cost_adjustment(A1, B, min_position)

        if E is not None:
            phi = potential_function(E + A1, B, min_position)
            is_min = is_global_min_enforced(phi, min_position)
            is_exact = is_valid_exact_potential(A1 + E, B, phi)

            print(is_min, is_exact)
            if is_min and is_exact and (np.linalg.norm(E) < np.linalg.norm(E_star)):
                print("Pos: ", min_position)
                E_star = E

    if np.any(np.isinf(E_star)):
        return None
    else:
        return E_star


if __name__ == '__main__':
    # ======= Test Matrices =======
    # A1 = np.array([[0, 1, 2],
    #                [-1, 0, 1],
    #                [-2, -1, 0]])
    # A2 = np.array([[0, 1, 2],
    #                [1, 2, 3],
    #                [2, 3, 4]])
    #
    # B = -A1+ 2*A2

    # size = 9
    # A1 = np.random.randint(0, 10, (size, size))  # Random integers from 0 to 9
    # B1 = np.random.randint(0, 10, (size, size))
    # A2 = np.random.randint(0, 10, (size, size))

    # A1 = np.array([[3, 1, 4, 1, 5, 8, 1 ,5, 7],
    #              [5, 3, 3, 2, 5, 8, 7, 1, 2],
    #              [5, 4, 6, 1, 0 ,7, 8, 5, 2],
    #              [0, 8, 2, 7, 6, 8, 6, 6, 6],
    #              [9, 7, 4, 5, 2, 4, 9, 4, 3],
    #              [0, 2, 7, 6, 4, 8, 5, 4, 4],
    #              [4, 9, 9, 5, 4, 6, 7, 9, 1],
    #              [9, 3, 4, 1, 6, 2, 2, 6, 7],
    #              [8, 6, 9, 7, 9, 7, 6, 1, 8]])
    #
    # A2 = np.array([[2, 7, 2, 6 ,1 ,3 ,5 ,9, 5],
    #              [1, 6, 9 ,0 ,1 ,2, 3, 8, 7],
    #              [3, 6, 8, 5, 5, 7, 3, 9, 2],
    #              [3, 0, 4, 0, 1, 3, 3, 7, 9],
    #              [4, 2, 5, 2, 7 ,9 ,1, 5, 9],
    #              [1 ,1 ,0 ,1, 7, 3, 0, 8, 6],
    #              [9, 3, 8, 3, 9, 0, 0, 5, 3],
    #              [8 ,5 ,8, 9, 5, 0, 8, 1, 4],
    #              [4, 5, 9, 4, 9, 6, 9, 1, 7]])
    #
    # B = np.array( [[9, 3, 5, 1, 3, 6, 6, 5, 5],
    #              [8, 7, 6, 3, 9, 0, 9, 3, 4],
    #              [8, 3, 1, 6, 4, 7, 8, 4, 6],
    #              [6, 5, 0, 5, 9, 9, 9, 3, 5],
    #              [0, 6, 7, 8, 7, 0, 2, 4, 0],
    #              [8, 9, 0 ,1, 3, 6, 4, 9, 1],
    #              [6, 0, 3, 7, 7, 5, 3, 1, 3],
    #              [9, 0, 0, 7, 7, 8, 1, 9, 9],
    #              [6, 3, 8, 2, 5, 7, 9, 4, 4]])

    A1_load = np.load('A1.npz')
    A2_load = np.load('A1.npz')
    B_load = np.load('A1.npz')

    # Extract the array
    A1 = A1_load['arr']
    A2 = A2_load['arr']
    B = B_load['arr']
    E = find_adjusted_costs(A1, A2, B)

    if E is None:
        print('None')

    else:
        A_prime = A1 + E
        player1_sec = np.argmin(np.max(A_prime, axis=1))
        player2_sec = np.argmin(np.max(B.transpose(), axis=1))

        min_position = (player1_sec, player2_sec)
        phi = potential_function(A_prime, B, min_position)

        print("Error:")
        print(E)
        print("Player 1 A_prime:")
        print(A_prime)
        print("Potential Function:")
        print(phi)
        print("Global Min:", int(min_position[0]), int(min_position[1]))
        print("Global Minimum Enforced:", is_global_min_enforced(phi, min_position))
        print("Exact Potential:", is_valid_exact_potential(A_prime, B, phi))
