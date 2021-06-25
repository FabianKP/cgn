


from enum import Enum
import numpy as np

from ..constraints import Constraint, combine_constraints

class Type(Enum):
    ls = "ls"
    lse = "lse"
    lsi = "lsi"
    lsei = "lsei"
    lsb = "lsb"


class LSProblem:

    def __init__(self, A, b, C=None, d=None, E=None, f=None, lb=None):
        """
        Represents the least-squares problem
        min_x ||Ax-b||^2 s.t. Cx=d, Ex>=f, lb<=x.
        """
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.E = E
        self.f = f
        self.lb = lb

        # if there are only bound constraints, the problem is of type lsb
        if self.lb is not None and self.C is None and self.E is None:
            self.type = Type("lsb")
        elif self.lb is not None:
            # otherwise, the bound constraints have to be translated into inequality constraints:
            C_l = np.identity(self.lb.size)
            lb_constraint = Constraint(dim=lb.size, mat=C_l, vec=lb)
            ineq_constraint = Constraint(dim=lb.size, mat=C, vec=b)
            combined_inequality_constraint = combine_constraints([lb_constraint, ineq_constraint])
            self.C = combined_inequality_constraint.mat
            self.d = combined_inequality_constraint.vec
            self.type = Type("lsi")
        elif self.C is None and self.E is None:
            # unconstrained
            self.type = Type("ls")
        elif self.C is not None and self.E is None:
            self.type = Type("lse")
        elif self.C is None and self.E is not None:
            self.type = Type("lsi")
        else:
            self.type = Type("lsei")