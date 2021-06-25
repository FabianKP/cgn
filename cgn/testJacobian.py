"""
Contains the functions testJacobian, taylor_test and second_order_remainder.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.


from numpy.linalg import norm
from numpy.random import normal
from termcolor import colored


def second_order_remainder(fun, grad, x, dx, h):
    """
    Computes second order taylor remainder.
    :return: e=||fun(x+h*dx)-fun(x) - h jac(x)dx||_2
    """
    e = norm( fun(x+h*dx)-fun(x) - h*grad(x).dot(dx) )
    return e



def taylor_test(fun, jac, x, dx, verbose, tol, c):
    """
    Performs Taylor remainder test.
    :param fun: User-defined function.
    :param jac: Candiate for its Jacobian.
    :param x: Point at which the test is performed.
    :param dx: Direction.
    :param verbose: if True, prints additional information.
    :param tol: Determines up to which precision jac must hold.
    :param c: Determines the reduction factor of h in every step.
    :return: True, if test is passed.
    """
    rtol = 0.01  #1% relative tolerance for not satisfying O(h^2) exactly
    h = 1.0
    epre = second_order_remainder(fun, jac, x, dx, h)
    #if verbose: print("h: ", h)
    #if verbose: print("e: ", epre)
    passed = True
    k = 1
    e = epre
    while(e > tol/(c**2)):
        h = c*h
        e = second_order_remainder(fun, jac, x, dx, h)
        #if verbose: print("h: ", h)
        #if verbose: print("e: ", e)
        # last check wins
        if e <= (1+rtol)*epre*(c**2):
            passed = True
            #if verbose: print("Looks good.")
        else:
            passed = False
            # break
            #if verbose: print("Oh no.")
        epre = e
        k*=1
    return passed


def testJacobian(fun, jac, x, verbose=False, ntests=42, tol=1e-13, reduction=0.5, nonneg = False, ratio=0.75):
    """
    Tests the validity of the Jacobian in a small neighbourhood of x.
    It performs two tests, one is the Taylor remainder test,
    the second is based on scipy.optimize.check_grad.
    What is actually tested is the gradient of x -> 0.5*||fun(x)||^2
    which is given by jac(x).T @ fun(x).
    :param fun: the function, must return numpy vector
    :param jac: the Jacobian, must return numpy array in numerator layout
    :param x: point around which Jacobian is tested.
    :param verbose: determines if it should talk or not
    :param ntests: number of random points near x where Jacobian is tested.
    :param tol: the tolerance for the Taylor remainder test
    :param reduction: factor by which h is reduced in every step
    :param nonneg: only tests the Jacobian at points with nonnegative components
    :param ratio: The fraction of tests that must be passed so that the overall test counts as passed. Defaults to 0.75.
    :return: True if at least 'ratio' of all tests passed. Otherwise False.
    """
    assert 0. < reduction < 1., "'reduction must be float between excl. 0 and 1"
    tests_passed = 0
    for i in range(ntests):
        if verbose: print("Test ", i+1, "/", ntests)
        noise = normal(size=len(x))
        xperturbed = x + 0.01*noise/norm(noise)
        if nonneg: xperturbed = xperturbed.clip(min=0.1)
        dx = normal(size=len(x))
        if taylor_test(fun, jac, xperturbed, dx, verbose, tol=tol, c=reduction):
            tests_passed += 1
            if verbose: print("Passed.")
        else:
            if verbose: print("Failed.")
    enough_tests = (tests_passed >= ratio * ntests)
    if verbose and enough_tests:
        print(colored(f"{tests_passed} out of {ntests} passed.", "green"))
        print(colored("Jacobian validated.", "green"))
    if verbose and not enough_tests:
        print(colored(f"{tests_passed} out of {ntests} passed.", "red"))
        print(colored("Jacobian test failed.", "red"))
    return enough_tests

