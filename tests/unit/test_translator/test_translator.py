
import numpy as np

from cgn.translator import Translator
from cgn.cnls_solve import CNLS, NullConstraint

from tests.unit.test_problem.problem_fixtures import three_parameter_problem, unconstrained_problem


def test_translate(three_parameter_problem):
    # Initialize translator
    translator = Translator(three_parameter_problem)
    # Translate dummy problem
    dummy_cnls = translator.translate()
    assert dummy_cnls.dim == three_parameter_problem.n
    assert isinstance(dummy_cnls, CNLS)


def test_modify_function(three_parameter_problem):
    translator = Translator(three_parameter_problem)
    cnls = translator.translate()
    testfun = three_parameter_problem.fun
    modified_fun = translator._modify_function(testfun)
    # Check that modified function can be called with concatenated vector
    x_conc = np.random.randn(cnls.dim)
    y = modified_fun(x_conc)
    m = three_parameter_problem.m
    assert y.size == m


def test_translate_equality_constraints(three_parameter_problem):
    translator = Translator(three_parameter_problem)
    translated_eqcon = translator._combine_constraints("eq")
    # Get the Jacobian of the translated eqcon
    x = np.zeros(translated_eqcon.dim)
    jac1 = translated_eqcon.jac(x)[:, :-1]
    # Get the Jacobian of the original constraint
    x1 = np.zeros(three_parameter_problem.shape[0])
    x2 = np.zeros(three_parameter_problem.shape[1])
    jac0 = three_parameter_problem.constraints[0].jac(x1, x2)
    assert np.isclose(jac1, jac0).all()


def test_translate_inequality_constraint(three_parameter_problem):
    translator = Translator(three_parameter_problem)
    translated_incon = translator._combine_constraints("ineq")
    n1 = three_parameter_problem.shape[0]
    # Get Jacobian of the original inequality constraint
    y = np.zeros(three_parameter_problem.constraints[1].dim)
    jac0 = three_parameter_problem.constraints[1].jac(y)
    # Get corresponding part of the Jacobian of the translated inequality constraint.
    x = np.zeros(translated_incon.dim)
    jac1 = translated_incon.jac(x)
    assert np.isclose(jac1[:, n1:-1], jac0).all()


def test_translate_unconstrained(unconstrained_problem):
    translator = Translator(unconstrained_problem)
    eqcon = translator._combine_constraints("eq")
    incon = translator._combine_constraints("ineq")
    assert isinstance(eqcon, NullConstraint)
    assert isinstance(incon, NullConstraint)


