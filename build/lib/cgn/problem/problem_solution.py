
import numpy as np
from typing import Sequence


class ProblemSolution:
    """
    Container for the solution of an optimization problem defined as instance of :py:class:`cgn.Problem`.
    """
    _minimizer_tuple: Sequence[np.ndarray]
    _precision: np.ndarray
    _cost: float
    _success: bool
    _niter: int

    def minimizer(self, pname: str) -> np.ndarray:
        """
        Returns the minimizer for the parameter of given name.
        """
        raise NotImplementedError

    @property
    def minimizer_tuple(self) -> Sequence[np.ndarray]:
        """
        The tuple of minimizers.
        """
        return self._minimizer_tuple

    @property
    def precision(self) -> np.ndarray:
        """
        The posterior precision matrix. Of shape (n,n), where n is the overall parameter dimension.
        """
        return self._precision

    @property
    def cost(self) -> float:
        """
        The minimum of the cost function.
        """
        return self._cost

    @property
    def success(self) -> bool:
        """
        `True`, if the iteration converged successfully. `False`, if the iteration stopped for other reasons.
        """
        return self._success

    @property
    def niter(self) -> int:
        """
        The number of iterations.
        """
        return self._niter
