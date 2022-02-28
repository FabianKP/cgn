
********
Tutorial
********

Mathematical background
=======================

``cgn`` stands for "**C**\onstrained **G**\auss-**N**\ewton". It is a generalization
of the well-known Gauss-Newton method to nonlinear least-squares problems
with constraints.

``cgn`` is able to solve optimization problems of the general form

.. math::
    \min_{\mathbf x} \quad & ||\mathbf Q \mathbf F(\mathbf x)||_2^2 + \beta ||\mathbf R(\mathbf x - \mathbf m)||_2^2\\
          s.t. \quad & \mathbf A \mathbf x = \mathbf b, \quad \mathbf C \mathbf x \geq \mathbf d, \quad
    \mathbf G(\mathbf x) = \mathbf 0, \quad \mathbf H(\mathbf x) \geq \mathbf 0,
    \quad \mathbf l \leq \mathbf x \leq \mathbf u, \\
    \text{where } & \mathbf x \in \mathbb{R}^d, \quad \mathbf F: \mathbb{R}^n \to \mathbb{R}^m, \mathbf Q\in \mathbb{R}^{m \times m},
    \beta \geq 0, \quad \mathbf R \in \mathbb{R}^{r \times n}, \quad \mathbf m \in \mathbb{R}^n,\\
    & \mathbf A \in \mathbb{R}^{c_1 \times n},
    \mathbf C \in \mathbb{R}^{c_2 \times n}, \quad \mathbf G: \mathbb{R}^n \to \mathbb{R}^{d_2}, \quad \mathbf H: \mathbb{R}^n \to
    \mathbb{R}^{d_1}, \quad \mathbf l, \mathbf u \in [-\infty, \infty]^n.

The functions :math:`\mathbf F`, :math:`\mathbf G` and :math:`\mathbf H` might be nonlinear, in which case
the user has to provide the analytic derivatives.

The constrained Gauss-Newton method solves the above problem by **sequential linearization**:
Given a *feasible* initial guess :math:`\mathbf x_0` (i.e. :math:`\mathbf x_0` has to satisfy all of the
constraints), it defines an iteration

.. math::
    \mathbf x_{k+1} = \mathbf x_k + h_k \Delta \mathbf x_k,

where the search direction :math:`\Delta \mathbf x_k` is determined
by solving the linearized version of the original problem,

.. math::
    \min_{\Delta \mathbf x} \quad & ||\mathbf Q (\mathbf F(\mathbf x_k) + \mathbf F'(\mathbf x) \Delta
    \mathbf x_k||_2^2 + \beta ||\mathbf R(\mathbf x_k + \Delta \mathbf x - \mathbf m)||_2^2\\
              s.t. \quad & \mathbf A \Delta \mathbf x = \mathbf 0, \quad \mathbf C \Delta \mathbf x \geq 0, \quad
        \mathbf G(\mathbf x_k) + \mathbf G'(\mathbf x_k) \Delta \mathbf x_k = \mathbf 0,
        \quad \mathbf H(\mathbf x_k) + \mathbf H'(\mathbf x_k) \Delta \mathbf x_k \geq \mathbf 0,
        \quad \mathbf l \leq \mathbf x_k + \Delta \mathbf x \leq \mathbf u.

The steplength :math:`h_k` is determined using a line-search filter developed by Wächter and Biegler. See

    Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for
    large-scale nonlinear programming. Mathematical programming, 106(1), 25-57.

Usage
=====

As a test, we will solve the following small
For setting up a problem, you need to create a :py:class:`cgn.Parameter` object.

.. code:: python

    import cgn
    x = cgn.Parameter(dim=d, name="x")

Template
========

The following template allows you to set up a generic nonlinear least-squares
problem with constraints. The free parameter :math:`\mathbf x \in \mathbb{R}^n` is separated
into two variables :math:`\mathbf x_1 \in \mathbb{R}^{n_1}` and :math:`\mathbf x_2 \in \mathbb{R}^{n_2}`,
where :math:`n = n_1 + n_2`.
This is just for convenience, as it allows to set constraints for these two variables separately.
Computationally, the result will be the same if you formulate the problem in terms of the concatenated
vector :math:`\mathbf x`. It should be self-explanatory how to adapt this template to use 3 or more separate
parameter vectors.

.. code:: python
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


