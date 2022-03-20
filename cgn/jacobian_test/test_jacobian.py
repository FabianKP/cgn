
from numpy.linalg import norm
from numpy.random import normal
from termcolor import colored


def _second_order_remainder(fun, grad, x, dx, h):
    """
    Computes second order taylor remainder.

    .. math::
        e = ||F(x+h \\Delta x)-F(x) - h jac(x) \\Delta x||_2

    :return: The remainder :math:`e`.
    """
    e = norm(fun(x+h*dx)-fun(x) - h*grad(x).dot(dx))
    return e


def _taylor_test(fun, jac, x, dx, tol, c):
    """
    Performs Taylor remainder tests.

    :param fun: User-defined function.
    :param jac: Candiate for its Jacobian.
    :param x: Point at which the tests is performed.
    :param dx: Direction.
    :param tol: Determines up to which precision jac must hold.
    :param c: Determines the reduction factor of h in every step.
    :return: True, if tests is passed.
    """
    # 1% relative tolerance for not satisfying O(h^2) exactly
    rtol = 0.01
    h = 1.0
    epre = _second_order_remainder(fun, jac, x, dx, h)
    passed = True
    k = 1
    e = epre
    while e > tol/(c**2):
        h = c*h
        e = _second_order_remainder(fun, jac, x, dx, h)
        # last check wins
        if e <= (1+rtol)*epre*(c**2):
            passed = True
        else:
            passed = False
        epre = e
        k *= 1
    return passed


def test_jacobian(fun: callable, jac: callable, x: callable, verbose=False, ntests=42, tol=1e-13, reduction=0.5,
                  nonneg: bool = False, ratio: float = 0.75):
    """
    Tests the validity of the Jacobian for a multivariable function :math:`F` in a small neighbourhood of a point
    :math:`x` using the Taylor remainder test.
    What is actually tested is the gradient of :math:`f(x) = 0.5*||F(x)||^2`
    which is given by :math:`J(x).T F(x)`.

    :param fun: The function :math:`F`, must take a vector of shape (n,) as input and return a vector of shape (m, ).
    :param jac: The supposed Jacobian :math:`J`. Must return numpy array of shape (m, n).
    :param x: The point at which the Jacobian is tested.
    :param verbose: If True, output is printed to the console during the test.
    :param ntests: Number of random points near x where Jacobian is tested.
    :param tol: The tolerance for the Taylor remainder tests.
    :param reduction: Factor by which h is reduced in every step
    :param nonneg: Only tests the Jacobian at points with nonnegative components
    :param ratio: The fraction of tests that must be passed so that the overall tests counts as passed.
        Defaults to 0.75.
    :return: True if at least ``ratio`` of all tests passed. Otherwise, False.
    """
    assert 0. < reduction < 1., "'reduction must be float between excl. 0 and 1"
    tests_passed = 0
    for i in range(ntests):
        if verbose:
            print("Test ", i+1, "/", ntests)
        noise = normal(size=len(x))
        xperturbed = x + 0.01*noise/norm(noise)
        if nonneg:
            xperturbed = xperturbed.clip(min=0.1)
        dx = normal(size=len(x))
        if _taylor_test(fun, jac, xperturbed, dx, tol=tol, c=reduction):
            tests_passed += 1
            if verbose:
                print("Passed.")
        else:
            if verbose:
                print("Failed.")
    enough_tests = (tests_passed >= ratio * ntests)
    if verbose and enough_tests:
        print(colored(f"{tests_passed} out of {ntests} passed.", "green"))
        print(colored("Jacobian validated.", "green"))
    if verbose and not enough_tests:
        print(colored(f"{tests_passed} out of {ntests} passed.", "red"))
        print(colored("Jacobian tests failed.", "red"))
    return enough_tests
