
from copy import deepcopy
from typing import List
import numpy as np
import scipy.linalg

from ..regularization_operator import RegularizationOperator
from .null_operator import NullOperator


class BlockOperator(RegularizationOperator):
    """
    Given a list of regularization operators :math:`P_1, \\ldots, P_l`, we form the block operator
    :math:`P = \\mathrm{diag}(P_1, P_2, ..., P_l)`.
    """
    def __init__(self, operator_list: List[RegularizationOperator]):
        """
        :param operator_list:
        """
        # Assert that not all operators in operator_list are null.
        all_null = self._all_ops_in_list_null(operator_list)
        assert not all_null
        # Continue
        self._operators = deepcopy(operator_list)
        self._n_split_positions, self._r_split_positions = self._get_split_positions()
        mat = self._concatenate_matrices()
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        See :py:attr:`RegularizationOperator.fwd`
        """
        if v.ndim == 1:
            # vector case
            v_list = np.split(v, self._r_split_positions, axis=0)
            res_list = []
            for op, vec in zip(self._operators, v_list):
                u = vec
                sol = op.fwd(u)
                res_list.append(sol)
            w = np.concatenate(res_list)
        else:
            # matrix case
            w = self._mat @ v
        return w

    def adj(self, v: np.ndarray) -> np.ndarray:
        """
        See :py:attr:`RegularizationOperator.adj`.
        """
        if v.ndim == 1:
            # vector case
            v_list = np.split(v, self._r_split_positions, axis=0)
            res_list = []
            for op, vec in zip(self._operators, v_list):
                u = vec
                sol = op.adj(u)
                res_list.append(sol)
            w = np.concatenate(res_list)
        else:
            # matrix case
            w = self._mat.T @ v
        return w

    # PROTECTED

    @staticmethod
    def _all_ops_in_list_null(operator_list: List[RegularizationOperator]) -> bool:
        """"
        :return:
            Returns True if all elements in operator_list are NullOperator's.
            Otherwise, returns False
        """
        all_null = True
        for op in operator_list:
            if not isinstance(op, NullOperator):
                all_null = False
        return all_null

    def _concatenate_matrices(self):
        """
        Computes mat and imat for the block operator
        :return: array_like, array_like
        """
        # First, concatenate both matrices as if the are no null operators.
        mat_list = []
        # During that, keep track of all the indices that do not correspond to null operators.
        running_row_mat = 0
        rows_mat = []
        for op in self._operators:
            mat_list.append(op.mat)
            if not isinstance(op, NullOperator):
                rows_mat.extend(range(running_row_mat, running_row_mat + op.mat.shape[0]))
            running_row_mat += op.mat.shape[0]
        mat = scipy.linalg.block_diag(*mat_list)
        # Then, remove the rows that correspond to null operators in mat and the columns that correspond to
        # null operators in imat.
        mat = mat[rows_mat, :]
        return mat

    def _get_split_positions(self):
        """
        Computes the positions at which the vector has to be split for component-wise computation of fwd and inv.
        :return: list[int], list[int]
        """
        n_split_positions = []
        r_split_positions = []
        i = 0
        j = 0
        # get all the positions in the vector where a new vector *starts*
        for op in self._operators[:-1]:
            r = op.rdim
            n = op.dim
            n_split_positions.append(i + r)
            r_split_positions.append(j + n)
            i += r
            j += n
        return n_split_positions, r_split_positions
