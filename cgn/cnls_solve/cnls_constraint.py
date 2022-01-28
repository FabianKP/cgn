
import numpy as np
from typing import Literal, Union


class CNLSConstraint:

    _fun: callable
    _jac: callable
    _dim: int
    _cdim: int

    def fun(self, x: np.ndarray) -> np.ndarray:
        return self._fun(x)

    def jac(self, x: np.ndarray) -> np.ndarray:
        return self._jac(x)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def cdim(self) -> int:
        return self._cdim


class ConcreteConstraint(CNLSConstraint):

    def __init__(self, dim: int, fun: callable, jac: callable):
        self._check_input(dim, fun, jac)
        self._dim = dim
        self._fun = fun
        self._jac = jac
        x0 = np.zeros(self._dim)
        y = self._fun(x0)
        self._cdim = y.size

    def _check_input(self, dim: int, fun: callable, jac: callable):
        assert dim > 0
        x0 = np.zeros(dim)
        y = fun(x0)
        cdim = y.size
        j = jac(x0)
        assert j.shape == (cdim, dim)


class NullConstraint(CNLSConstraint):

    def __init__(self, dim: int):
        assert dim > 0
        self._dim = dim
        def null_fun(x: np.ndarray) -> np.ndarray:
            return np.zeros(0)  # Returns the empty array
        def null_jac(x: np.ndarray) -> np.ndarray:
            return np.zeros((0, dim))   # Returns empty array of correct shape!
        self._fun = null_fun
        self._jac = null_jac
        self._cdim = 0


