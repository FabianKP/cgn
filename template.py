import cgn


######## ADAPT THESE:


# Set the dimensions of the two parameters x_1 and x_2:
n_1 = ...
n_2 = ...

# Implement the function F.
def F(x_1, x_2):
    ...

# Implement the Jacobian of F.
def F_jacobian(x_1, x_2):
    ...

# Define the matrix Q
Q = ...

# Implement the function G that determines the nonlinear equality constraint G(x) = 0.
def G(x_1, x_2):
    ...

# Implement the Jacobian of G.
def G_jacobian(x_1, x_2):
    ...

# Implement the function H that determines the nonlinear inequality constraint H(x) >= 0.
def H(x_1, x_2):
    ...

# Implement the Jacobian of H.
def H_jacobian(x_1, x_2):
    ...

# Set the linear equality constraint A x = b.
A = ...
b = ...
# Set the linear inequality constraint C x >= d.
C = ...
d = ...
# Set the regularization terms
# ... for x_1:
beta_1 = ...
R_1 = ...
# ... for x_2:
beta_2 = ...
R_2 = ...

# Set the lower and upper bound constraints
# ... for x_1:
lb_1 = ...
ub_1 = ...
# ... for x_2:
lb_2 = ...
ub_2 = ...

# Set initial guesses for x_1 and x_2 (mandatory!)
x_1_guess = ...
x_2_guess = ...


######## DO NOT ADAPT:


# Initialize cgn.Parameter objects
x_1 = cgn.Parameter(dim=n_1, name="x1")
x_2 = cgn.Parameter(dim=n_2, name="x2")
# Set the regularization terms.
x_1.beta = beta_1
x_1.regop = R_1
x_2.beta = beta_2
x_2.regop = R_2
# Set the bound constraints.
x_1.lb = lb_1
x_1.ub = ub_1
x_2.lb = lb_2
x_2.ub = ub_2
# Initialize the constraint objects
linear_equality = cgn.LinearConstraint(parameters=[x_1, x_2], a=A, b=b, ctype="eq")
linear_inequality = cgn.LinearConstraint(parameters=[x_1, x_2], a=C, b=d, ctype="ineq")
nonlinear_equality = cgn.NonlinearConstraint(parameters=[x_1, x_2], fun=G, jac=G_jacobian, ctype="eq")
nonlinear_inequality = cgn.NonlinearConstraint(parameters=[x_1, x_2], fun=H, jac=H_jacobian, ctype="ineq")
constraint_list = [linear_equality, linear_inequality, nonlinear_equality, nonlinear_inequality]

# Finally, create a cgn.Problem object
optimization_problem = cgn.Problem(parameters=[x_1, x_2],
                                   fun=F,
                                   jac=G,
                                   q=Q,
                                   constraints=constraint_list)

# Create a cgn.CGN solver object
solver = cgn.CGN()

# Solve the problem
solution = solver.solve(problem=optimization_problem, starting_values=[x_1_guess, x_2_guess])

# Now, the solutions can be accessed via
x_1_solution = solution.minimizer("x1")
x_2_solution = solution.minimizer("x2")





