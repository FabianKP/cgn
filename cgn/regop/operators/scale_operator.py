
from ..regularization_operator import RegularizationOperator
from .null_operator import NullOperator
from .scaled_operator import ScaledOperator


def scale_operator(regop: RegularizationOperator, alpha: float) -> RegularizationOperator:
    """
    Scales a regularization operator: Given a regularization operator :math:`P` and a constant :math:`\alpha`,
    the new operator is :math:`\\sqrt(\\alpha) P`.
    :param regop:
    :param alpha:
    :returns: If `alpha` is close to 0, then the returned operator is a :py:class:`NullOperator`. Otherwise, it is a
    :py:class:`ScaledOperator`.
    """
    if abs(alpha) < 1e-20:  # if alpha=0
        scaled_operator = NullOperator(dim=regop.dim)
    else:
        scaled_operator = ScaledOperator(regop=regop, alpha=alpha)
    return scaled_operator