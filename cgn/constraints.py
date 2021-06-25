"""
Contains classes: NonlinearConstraint, EqualityConstraint, InequalityConstraint, BoundConstraint
"""

import numpy as np
import scipy.linalg


class Constraint:
    """
    Represents the constraint mat @ x >=/= vec
    """
    def __init__(self, dim, mat=None, vec=None):
        if mat is None:
            # empty constraint
            self.dim = dim
            self.cdim = 0
            self.mat = np.zeros((1, dim))
            self.vec = np.zeros(1)
            self.empty = True
        else:
            assert mat.shape[1] == dim
            self.dim = dim
            self.cdim = mat.shape[0]
            self.mat = mat
            self.vec = vec
            self.empty = False


def combine_constraints(constraint_list):
    """
    Given a list of constraints, returns a single constraint for the concatenated vector.
    :param constraint_list: list
        List of objects of type constraint
    :return: Constraint
    """
    # build list of all constraint matrices.
    # keep track whether there all constraints are None, since then we have to build a None constraint
    matrices = []
    vectors = []
    combined_dim = 0
    all_none = True
    for constraint in constraint_list:
        matrices.append(constraint.mat)
        vectors.append(constraint.vec)
        combined_dim += constraint.dim
        if constraint.cdim > 0:
            all_none = False
    # if all constraints are None, return a None constraint of the right dimension
    if all_none:
        combined_constraint = Constraint(dim=combined_dim)
    else:
        mat = scipy.linalg.block_diag(*matrices)
        vec = np.concatenate(vectors)
        combined_constraint = Constraint(dim=combined_dim, mat=mat, vec=vec)
    return combined_constraint
