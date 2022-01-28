
import numpy as np

from ..problem.constraint import Constraint


def get_sub_matrix(mat: np.ndarray, constraint: Constraint, i: int) -> np.ndarray:
    """
    Returns the sub-matrix of the constraint that corresponds to the i-th parameter in the constraint.
    """
    # Check that mat has the correct dimensions.
    assert mat.shape == (constraint.cdim, constraint.dim)
    # Compute the position of the parameter in the concatenated parameter vector
    parameters = constraint.parameters
    if i not in range(len(parameters)):
        raise Exception(f"'i' must be between 0 and {len(parameters)}")
    pos_i = 0
    for j in range(i):
        pos_i += parameters[j].dim
    dim_i = parameters[i].dim
    # Get the sub-matrix
    mat_sub = mat[:, pos_i:pos_i + dim_i]
    return mat_sub