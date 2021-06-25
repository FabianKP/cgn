"""
Contains 3 tests for the BGN method.
"""

from termcolor import colored
from time import time

from testOneDimensional import testOneDimensional
from testLinear import testLinear
from testOsborne import osborneTest
from test_fully_constrained import test_fully_constrained
from test_linear_constrained import test_linear_constrained
from test_osborne_bounds import test_osborne_bounds


# main
tests = [testOneDimensional, testLinear, test_linear_constrained, test_fully_constrained, osborneTest, test_osborne_bounds]
excludeTests = [test_osborne_bounds]
testnames = ["One-dimensional test", "Linear test", "Linear test with equality constraint",
             "Nonlinear test with equality and inequality constraints", "Osborne test", "Osborne test with lower bounds"]
allTestsPassed = True
result = {}
f = open("benchmark_Cgn_0.0.1.log", "w")
for test, name in zip(tests, testnames):
    if test not in excludeTests:
        t0 = time()
        result[name]=test()
        t_test = time()-t0
        print(f"{name} took {t_test} seconds.")
        f.write(f"{name} took {t_test} seconds.\n")
f.close()
for name in result.keys():
    if result[name]:
        print(colored(name, "green"))
    else:
        print(colored(name, "red"))
