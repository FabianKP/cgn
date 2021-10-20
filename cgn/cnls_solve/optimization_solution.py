
from dataclasses import dataclass
from enum import Enum
from numpy.typing import ArrayLike


class OptimizationStatus(Enum):
    converged = "converged"
    timeout = "timeout"
    maxout = "maxout"
    error = "error"


@dataclass(frozen=True)
class OptimizationSolution:
    minimizer: list
    precision: ArrayLike
    min_cost: float
    status: OptimizationStatus
    niter: int