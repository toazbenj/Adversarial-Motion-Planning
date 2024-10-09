"""
Matrix Cleaning

Testing functions to get rid of numpy nan values (impossible states) so that matrices can be passed into regular
NE solvers
"""


# https://stackoverflow.com/questions/11620914/how-do-i-remove-nan-values-from-a-numpy-array
import numpy as np

def clean_matrix3(mat):
    # Remove rows/cols with all NaNs
    mat = mat[~np.isnan(mat).all(axis=1)]
    mat = mat[:, ~np.isnan(mat).all(axis=0)]
    return mat

def clean_matrix2(mat):
    # Remove rows with all NaNs
    mat = mat[~np.isnan(mat).all(axis=1)]

    # Initialize a clean matrix with NaNs
    clean_mat = np.full_like(mat, np.nan)

    # Shift non-NaN values to the left
    for i in range(mat.shape[0]):
        nan_count = 0
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                nan_count += 1
            else:
                clean_mat[i, j - nan_count] = mat[i, j]

    # Remove columns that contain any NaNs
    clean_mat = clean_mat[:, ~np.isnan(clean_mat).any(axis=0)]

    return clean_mat


def clean_matrix(mat):
    clean_mat = mat.copy()
    clean_mat.fill(float('nan'))

    for i in range(mat.shape[0]):
        nan_count = 0
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                nan_count += 1
            else:
                clean_mat[i, j-nan_count] = mat[i][j]

    clean_mat = clean_mat[:, ~np.isnan(clean_mat).any(axis=0)]
    return clean_mat


def remap_values(mat, small_arr, is_row=True):
    """
    Map values from small arr to large arr, each large arr index corresponds to non nan value
    in either row or col of mat
    :param mat: mat of cost values
    :param small_arr: list of probabilities from cleaned payoff matrix
    :param is_row: policies for row or col
    :return: arr with non nan value indexes with probabilities, rest set to 0
    """
    large_arr = np.zeros((len(mat)))
    small_lst = list(small_arr)
    if is_row:
        mapping_arr = np.isfinite(mat[:, ~np.isnan(mat).all(axis=0)])[:,0]
    else:
        mapping_arr = np.isfinite(mat[~np.isnan(mat).all(axis=1)])[0]

    for i in range(len(large_arr)):
        if mapping_arr[i]:
            large_arr[i] = small_lst.pop(0)
        i += 1
    return large_arr

# x = np.array([[1, float('nan'), 3],
#               [4, float('nan'), 6],
#               [float('nan'), float('nan'), float('nan')]])

x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,  1., np.nan,  0.,  1., np.nan, np.nan, np.nan],
       [np.nan, np.nan,  1., np.nan, -1.,  0., np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,  1., np.nan, -1.,  1., np.nan, np.nan, np.nan]])

# x = x[~np.isnan(x).any(axis=1)]
# x = x[~np.isnan(x)]

# # first get the indices where the values are finite
# ii = np.isfinite(x)
#
# # second get the values
# x = x[ii]

# x = clean_matrix3(x)
# print(x)

small_arr = np.array([0.2, 0.7, 0.1])
policy1 = remap_values(x, small_arr)
policy2 = remap_values(x, small_arr, is_row=False)
print("policy1 = ", policy1)
print("policy2 = ", policy2)
