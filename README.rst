CGN
============

**CGN** stands for "Constrained Gauss-Newton method".
This package contains a **Python** implementation of the Gauss-Newton method for nonlinear least-squares
problems with linear constraints. That is, CGN provides a method for solving constrained optimization
problems of the form
    .. math::
        \\min_{x_1,...,x_p} \qquad & ||Q F(x_1,...,x_p)||_2^2 + \beta_1 * ||R_1(x_1 - m_1)||_2^2 + \ldots +
        \beta_2 * ||R_p(x_p - m_p)||_2^2 \\\\
          \text{subject to} \qquad & Ax = b, \quad Cx \geq d, \quad G(x) = 0, H(x) \geq 0, \quad l \leq x \leq u,
        x = \left(\begin{matrix} x_1 \\ \vdots \\ x_p \end{matrix} \right).


Installation
============

For the installation, you need to have

- Python 3, and
- pip

installated on your computer. **CGN** depends on the following packages,
which will be installed automatically during installation:

- numpy
- scipy
- qpsolvers
- typing
- logging
- prettytable


Tutorial
========
