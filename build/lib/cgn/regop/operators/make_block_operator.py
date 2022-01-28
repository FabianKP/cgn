
from typing import List

from ..regularization_operator import RegularizationOperator
from .block_operator import BlockOperator
from .null_operator import NullOperator


def make_block_operator(operator_list: List) -> RegularizationOperator:
    """
    Given a list of regularization operators, creates a block operator as a direct sum.
    :param operator_list:
    :return: The resulting operator might either be a :py:class:`BlockOperator', or a :py:class:`NullOperator`
    if all operators in the list are of instances of :py:class:`NullOperator`.
    """
    # Check if all operators in the list are null.
    all_null = True
    for op in operator_list:
        if not isinstance(op, NullOperator):
            all_null = False
    # If yes, return a NullOperator of the right dimension.
    if all_null:
        # If yes, return a NullOperator of the right dimension.
        combined_dim = 0
        for op in operator_list:
            combined_dim += op.dim
        block_operator = NullOperator(combined_dim)
    # If not, return a BlockOperator.
    else:
        block_operator = BlockOperator(operator_list)
    return block_operator
