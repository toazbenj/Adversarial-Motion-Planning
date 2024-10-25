import numpy as np

# Function to map values from small matrix to large matrix

def remap_matrix(example_mat, small_mat):
    """
    Map values from small matrix into larger example matrix. Real-valued entries in example_mat
    are replaced by corresponding values from small_mat, while other entries are set to NaN.

    :param example_mat: Large matrix with some entries as real values and others as NaN
    :param small_mat: Smaller matrix with values to fill into real-valued positions of example_mat
    :return: Matrix with real-valued positions from example_mat filled with values from small_mat, others set to NaN
    """
    # Ensure example_mat is flattened
    flat_example = example_mat.flatten()
    # Create an output matrix of the same shape, defaulting to NaN
    output_mat = np.full_like(flat_example, np.nan, dtype=np.float64)

    # Extract positions of finite (real) values in example_mat
    real_positions = np.isfinite(flat_example)

    # Check if the number of real positions matches the length of small_mat
    if real_positions.sum() != len(small_mat):
        raise ValueError("Number of non-NaN values in example_mat does not match the size of small_mat")

    # Fill real positions in the output matrix with values from small_mat
    output_mat[real_positions] = small_mat.flatten()

    # Reshape to the original dimensions of example_mat
    return output_mat.reshape(example_mat.shape)


# Example input matrices
example_mat = np.array([
    [np.nan, 2.0, np.nan],
    [3.0, np.nan, 4.0],
    [np.nan, 5.0, np.nan]
])

small_mat = np.array([0.1, 0.2, 0.3, 0.4])
# Test the function
output_matrix = remap_matrix(example_mat, small_mat)
print(output_matrix)
