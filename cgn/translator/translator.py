
import numpy as np
from typing import List, Literal

from ..cnls_solve import CNLS, CNLSConstraint, ConcreteConstraint, NullConstraint
from ..cnls_solve.cnls_solution import CNLSSolution, OptimizationStatus

from ..problem.linear_constraint import Constraint
from ..translator.get_sub_matrix import get_sub_matrix
from ..problem.problem import Problem
from .multiparameter import MultiParameter
from .translated_solution import TranslatedSolution


class Translator:
    """
    Translates a cgn.Problem object to a CNLS object.
    """
    def __init__(self, problem: Problem):
        self._problem = problem
        self._nparams = problem.nparams
        # read the problem parameters into a Parameters object
        self._multi_parameter = MultiParameter(self._problem.parameters)

    def translate(self) -> CNLS:
        """
        Returns a CNLS equivalent to the :py:class``Problem`` object.
        """
        fun = self._modify_function(self._problem.fun)
        jac = self._modify_function(self._problem.jac)
        q = self._problem.q
        eqcon = self._combine_constraints(ctype="eq")
        incon = self._combine_constraints(ctype="ineq")
        mean = self._multi_parameter.mean
        r = self._multi_parameter.regop
        lb = self._multi_parameter.lb
        ub = self._multi_parameter.ub
        scale = self._problem.scale
        cnls = CNLS(func=fun, jac=jac, q=q, r=r, m=mean, eqcon=eqcon, incon=incon, lb=lb, ub=ub, scale=scale)
        return cnls

    def _modify_function(self, func: callable):
        """
        Takes function that takes list of arguments and transforms it to function that takes concatenated
        vector as input.
        :param func: function that takes a tuple as argument
        :return: function that takes a single vector as argument
        """
        def newfunc(x):
            x_tuple = self._extract_x(x)
            return func(*x_tuple)
        return newfunc

    def _extract_x(self, x):
        """
        From a concatenated vector, extracts the tuple of parameters
        """
        return self._multi_parameter.split(x)

    def combine_x(self, x_list):
        assert len(x_list) == self._nparams
        return np.concatenate(x_list)

    def _combine_constraints(self, ctype: Literal["eq", "ineq"]) -> CNLSConstraint:
        """
        Reads all constraints of the given type from self.problem and returns one constraint
        for the concatenated vector. Might be the null constraint.
        """
        # Get all constraints of given ctype from self._problem as list.
        constraint_list = self._get_constraints(ctype=ctype)
        # If the list is empty, return a NullConstraint.
        if len(constraint_list) == 0:
            combined_constraint = NullConstraint(dim=self._problem.n)
        # Else, return the concatenated constraint.
        else:
            # First, we have to formulate all constraints with respect to the concatenated parameter vector.
            list_of_enlarged_constraints = []
            for constraint in constraint_list:
                enlarged_constraint = self._enlarge_constraint(constraint=constraint)
                list_of_enlarged_constraints.append(enlarged_constraint)
            # Then, we merge the constraints
            combined_constraint = self._concatenate_constraints(list_of_enlarged_constraints)
        return combined_constraint

    def _concatenate_constraints(self, list_of_constraints: List[ConcreteConstraint]) -> ConcreteConstraint:
        """
        Given a list of :py:class:`ConcreteConstraint` objects, returns a ConcreteConstraint that represents the
        concatenated constraint.
        """
        # Define the concatenated function.
        def concatenated_fun(x: np.ndarray):
            y_list = []
            for con in list_of_constraints:
                y = con.fun(x)
                y_list.append(y)
            y = np.hstack([y_list]).flatten()   # Has to flatten, otherwise input might be (1,1) instead of (1,).
            return y

        def concatenated_jac(x: np.ndarray):
            j_list = []
            for con in list_of_constraints:
                j = con.jac(x)
                j_list.append(j)
            j = np.concatenate(j_list, axis=0)
            return j
        concatenated_constraint = ConcreteConstraint(dim=self._multi_parameter.dim, fun=concatenated_fun,
                                                     jac=concatenated_jac)
        return concatenated_constraint

    def _get_constraints(self, ctype: Literal["eq", "ineq"]) -> List[Constraint]:
        """
        Returns all constraints of self._problem with the given ctype.
        """
        constraint_list = []
        for constraint in self._problem.constraints:
            if constraint.ctype == ctype:
                constraint_list.append(constraint)
        return constraint_list

    def _enlarge_constraint(self, constraint: Constraint) -> ConcreteConstraint:
        """
        Given an object of type :py:class:`Constraint`, returns the equivalent object of type
        :py:class:`ConcreteConstraint` formulated with respect to the concatenated parameter vector.
        """
        # Define the concatenated constraint function.
        enlarged_fun = self._enlarge_function(constraint)
        enlarged_jac = self._enlarge_jacobian(constraint)
        # Create a ConcreteConstraint object from a_enlarged
        concrete_constraint = ConcreteConstraint(dim=self._problem.n, fun=enlarged_fun, jac=enlarged_jac)
        return concrete_constraint

    def translate_solution(self, cnls_solution: CNLSSolution) -> TranslatedSolution:
        """
        Translates the solution of the CNLS problem to the solution of the original, multi-parameter problem
        """
        xmin = self._multi_parameter.split(cnls_solution.minimizer)
        precision = cnls_solution.precision
        cost = cnls_solution.min_cost
        niter = cnls_solution.niter
        success = (cnls_solution.status == OptimizationStatus.converged)
        problem_solution = TranslatedSolution(parameters=self._problem.parameters, minimizers=xmin,
                                              precision=precision, cost=cost, success=success, niter=niter)
        return problem_solution

    def _enlarge_function(self, constraint: Constraint) -> callable:
        """
        Given a function that depends on some list of parameters, return the equivalent function that takes the
        concatenated parameter vector as input.

        :param constraint:
        """
        parameters = constraint.parameters
        function = constraint.fun

        def enlarged_fun(x: np.ndarray):
            # Split x into parameters.
            x_list = self._multi_parameter.split(x)
            # Call the original function with the corresponding parameters.
            indices = self._multi_parameter.get_indices(parameters)
            args = [x_list[i] for i in indices]
            return function(*args)
        return enlarged_fun

    def _enlarge_jacobian(self, constraint: Constraint) -> callable:
        """
        Given a Jacobian that depends on some list of parameters, return the equivalent Jacobian that takes the
        concatenated parameter vector as input.

        :param constraint:
        :return:
        """
        jacobian = constraint.jac
        parameters = constraint.parameters

        def enlarged_jac(x: np.ndarray):
            # Split x into parameter list
            x_list = self._multi_parameter.split(x)
            # Call the original Jacobian with the corresponding parameters.
            indices = self._multi_parameter.get_indices(parameters)
            args = [x_list[i] for i in indices]
            j = jacobian(*args)
            # Enlarge the Jacobian array so that it has the right dimensions.
            j_enlarged = self._enlarge_matrix(j, constraint)
            # Return the enlarged array.
            return j_enlarged
        return enlarged_jac

    def _enlarge_matrix(self, mat: np.ndarray, constraint: Constraint):
        a_enlarged = np.zeros((constraint.cdim, self._problem.n))
        parameters = constraint.parameters
        # Write the splitted matrices in the enlarged matrix at the right positions
        for i in range(len(parameters)):
            a_i = get_sub_matrix(mat, constraint, i)
            name = parameters[i].name
            j_i = self._multi_parameter.position_by_name(name)
            k_i = a_i.shape[1]
            a_enlarged[:, j_i:j_i + k_i] = a_i
        return a_enlarged
