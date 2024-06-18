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

x = np.array([[1, float('nan'), 3],
              [4, float('nan'), 6],
              [float('nan'), float('nan'), float('nan')]])

# x = x[~np.isnan(x).any(axis=1)]
# x = x[~np.isnan(x)]

# # first get the indices where the values are finite
# ii = np.isfinite(x)
#
# # second get the values
# x = x[ii]

x = clean_matrix3(x)

print(x)
