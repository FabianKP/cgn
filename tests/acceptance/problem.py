"""
Contains class "TestProblem"
"""
import cgn
from cgn.cnls_solve.solveroptions import Solveroptions


class TestProblem:
    """
    Base class for test problems.
    """
    def __init__(self):
        self._problem = None
        self._minimum = None
        self._tol = None
        self._options = Solveroptions()
        self._options.set_verbosity(1)
        self._start = None

    @property
    def cgnproblem(self) -> cgn.Problem:
        return self._problem

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def options(self) -> Solveroptions:
        return self._options

    @property
    def start(self):
        return self._start