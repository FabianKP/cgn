
from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass(frozen=True)
class ProblemSolution:
    """
    Container for the solution of an optimization problem defined as instance of :py:class:`cgn.Problem`.

    :ivar minimizer: The minimizer of the optimization problem, given as a list [x0, x1, ...].
    :ivar precision: The posterior precision matrix. Of shape (n,n), where n is the overall parameter dimension.
    :ivar cost: The minimum of the cost function.
    :ivar success: `True`, if the iteration converged successfully. `False`, if the iteration stopped for other reasons.
    :ivar niter: The number of iterations.
    """
    minimizer: List[np.ndarray]
    precision: np.ndarray
    cost: float
    success: bool
    niter: int
