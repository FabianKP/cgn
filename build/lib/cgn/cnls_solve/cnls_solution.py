
from dataclasses import dataclass
from enum import Enum
import numpy as np


class OptimizationStatus(Enum):
    converged = "converged"
    timeout = "timeout"
    maxout = "maxout"
    error = "error"
    constraint_violated = "constraint violated"


@dataclass(frozen=True)
class CNLSSolution:
    minimizer: np.ndarray
    precision: np.ndarray
    min_cost: float
    status: OptimizationStatus
    niter: int