
********
Tutorial
********

Examples
--------

For various example applications of ``cgn``, see the `examples <https://github.com/FabianKP/cgn/tree/main/examples>`_
folder.

Defining parametes
------------------

The free parameter :math:`\mathbf x` has to be modelled as a :py:class:`cgn.Parameter` object. During initialization the
user has to provide a name for the parameter and a first guess. Note that this guess has to be a feasible point, i.e.
it has to satisfy all the constraints of your optimization problem (see below).

For example, the following code declares a parameter ``x`` of dimension 12 with name "x" and starting guess equal to
the zero vector.

.. code-block:: python

    import cgn
    import numpy as np

    x = cgn.Parameter(name="x", start=np.zeros(12))

In a lot of applications, your free parameter :math:`\mathbf x` will actually be a concatenation
of various one- or multi-dimensional parameters that deal with different aspects of your model.

For example, in time series regression it might be the case that :math:`\mathbf x` can be written as

.. math::
    \mathbf x = \left( \begin{matrix} \mathbf z \\ \mathbf \theta \end{matrix} \right),

where :math:`\mathbf z \in \mathbb R^{n_z}` corresponds to the values of your time series, and
:math:`\mathbf \theta \in \mathbb R^{n_\theta}`
corresponds to some hyperparameters of your model.
In that case, it might be more convenient do define the function :math:`\mathbf F` or the
various constraints directly in terms of :math:`\mathbf z` and :math:`\mathbf \theta`.
This can be achieved in ``cgn``, which lets you define your problem in terms of
multiple :py:class:`cgn.Parameter` objects.

.. autoclass: cgn.Parameter


Setting up optimization problems
--------------------------------

In order to use ``cgn``, you have to create a :py:class:`cgn.Problem` object that
represents your optimization problem.

.. autoclass:: cgn.Problem
    :noindex:
    :members:


Defining linear constraints
---------------------------

Linear constraints of the form :math:`\mathbf A \mathbf x = \mathbf b` or :math:`\mathbf C \mathbf
x \geq \mathbf d`, have to be defined as :py:class:`cgn.LinearConstraint` objects.

.. autoclass:: cgn.LinearConstraint
    :noindex:
    :members:

Defining nonlinear constraints
------------------------------

Nonlinear constraints of the form :math:`\mathbf G(\mathbf x) = \mathbf 0` or
:math:`\mathbf H(\mathbf x) \geq \mathbf 0` have to be defined as :py:class:`cgn.NonlinearConstraint` objects.

.. automodule:: cgn.NonlinearConstraint
    :noindex:
    :members:

Defining bound constraints
--------------------------

In order to define a bound constraint of the form :math:`\mathbf l \leq \mathbf x \leq \mathbf u` on a parameter
:math:`\mathbf x`, simply change the attributes of corresponding :py:class:`cgn.Parameter` object.

As an example, the following code defines a 3-dimensional parameter :math:`\mathbf x` with bound constraints
:math:`0 \leq x_1 \leq 1` and :math:`0 \leq x_2`:

.. code-block:: python

    import numpy as np

    x = cgn.Parameter(name="x", start=np.zeros(3))
    x.lb = np.array([0., 0., -np.inf])
    x.ub = np.array([1., np.inf, np.inf])

Defining regularization terms
-----------------------------

When you set up a parameter :math:`\mathbf x \in \mathbb R^n` in ``cgn``, it automatically creates a regularization term of the form
:math:`\beta ||\mathbf R (\mathbf x - \mathbf m)||` with default values

.. math::
    \beta & = 0, \\
    \mathbf R & = \mathbf{I}_n, \\
    \mathbf m & = \mathbf 0,

where :math:`\mathbf I_n \in \mathbb R^{n \times n}` denotes the identity matrix. Because :math:`\beta = 0`,
this regularization term has no effect. You can change the regularization term by adapting the corresponding
attributes of the :py:class:`cgn.Parameter` object. Note that this can even be done after the `cgn.Problem`-object has
been initialized. For example, you can create a problem, solve it, then adapt some parameters, and solve it again,
without having to create two different `cgn.Problem` objects.

The following code sets up an `n`-dimensional
parameter with regularization term :math:`0.3\cdot ||\mathbf R(\mathbf x - \mathbf 1_n)||`, where
:math:`\mathbf R = \mathrm{diag}(1, \frac{1}{2}, \ldots, \frac{1}{n})`:

.. code-block:: python

    import numpy as np

    x = cgn.Parameter(name="x", start=np.zeros(n))
    x.beta = 0.3
    x.mean = np.ones(n)
    R = np.diag(np.divide(np.ones(n), np.arange(1, n+1)))
    x.regop = R

The regularization operator :code:`x.regop` can be either a 2-dimensional numpy array, or a
:py:class:`cgn.RegularizationOperator`.

.. automodule:: cgn.RegularizationOperator
    :noindex:
    :members:

Setting up the solver
---------------------

Before you can solve your optimization problem, you need to create a :py:class:`cgn.CGN` object.
This represent the solver.

.. code-block:: python

    solver = cgn.CGN()

.. autoclass:: cgn.CGN
    :noindex:
    :members:

You can then set the solver options by modifying the :py:attr:`cgn.CGN.solveroptions` attribute.
As an example, the following code changes the maximum number of iterations to 300:

.. code-block:: python

    solver.options.maxiter = 300

.. autoclass:: cgn.Solveroptions
    :noindex:
    :members:

Solving the optimization problem
--------------------------------

Suppose you have successfully set up a :py:class:`cgn.Problem` object ``myproblem`` and a
:py:class:cgn.CGN` object ``solver``. As a last step, you need to set up the starting values
for the parameters. This must be a list of the same length as ``myproblem.parameters``,
and each element is a numpy vector that defines the starting value for the corresponding parameter.

Note that if your problem is constrained, then the starting values must satisfy these constraints.

Afterwards, you can call the solver using the code

.. code-block:: python

    solution = solver.solve(problem=my_problem)

The output `solution` is an object of type :py:class:`cgn.ProblemSolution`.

.. autoclass:: cgn.ProblemSolution
    :noindex:
    :members:

Template
========

If you want to quickly set up your problem in ``cgn``, you can copy and adapt the following template
that defines a generic nonlinear least-squares
problem with constraints. The free parameter :math:`\mathbf x \in \mathbb{R}^n` is separated
into two variables :math:`\mathbf x_1 \in \mathbb{R}^{n_1}` and :math:`\mathbf x_2 \in \mathbb{R}^{n_2}`,
where :math:`n = n_1 + n_2`.
This is just for convenience, as it allows to set constraints for these two variables separately.
Computationally, the result will be the same if you formulate the problem in terms of the concatenated
vector :math:`\mathbf x`. It should be self-explanatory how to adapt this template to 3 or more separate
parameters.

.. code-block:: python

    import cgn


    ######## PROBLEM DEFINITION

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


    ######## SET UP CGN:


    # Initialize cgn.Parameter objects
    x_1 = cgn.Parameter(name="x1", start=x_1_guess)
    x_2 = cgn.Parameter(name="x2", start=x_2_guess)
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

    # You can configure the solver by adapting `solver.options`.
    # For example, set the maximum number of iterations:
    solver.options.maxiter = ...

    # Solve the problem
    solution = solver.solve(problem=optimization_problem, starting_values=[x_1_guess, x_2_guess])

    # Now, the solutions can be accessed via
    x_1_solution = solution.minimizer("x1")
    x_2_solution = solution.minimizer("x2")


