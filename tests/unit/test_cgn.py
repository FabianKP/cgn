
from cgn import CGN


def test_cgn():
    solver = CGN()
    solver.options.tol = 1e-4
    solver.options.maxiter = 1000
    solver.linesearch.eta = 0.1