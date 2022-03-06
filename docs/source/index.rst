.. cgn documentation master file, created by
   sphinx-quickstart on Wed Oct 20 12:43:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*******************************
CGN
*******************************

``cgn`` stands for "**C**\onstrained **G**\auss-**N**\ewton". It is a generalization
of the well-known Gauss-Newton method to nonlinear least-squares problems
with constraints.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   tutorial.rst

Mathematical background
-----------------------

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
    \mathbf x_k)||_2^2 + \beta ||\mathbf R(\mathbf x_k + \Delta \mathbf x - \mathbf m)||_2^2\\
              s.t. \quad & \mathbf A \Delta \mathbf x = \mathbf 0, \quad \mathbf C \Delta \mathbf x \geq 0, \quad
        \mathbf G(\mathbf x_k) + \mathbf G'(\mathbf x_k) \Delta \mathbf x_k = \mathbf 0,
        \quad \mathbf H(\mathbf x_k) + \mathbf H'(\mathbf x_k) \Delta \mathbf x_k \geq \mathbf 0,
        \quad \mathbf l \leq \mathbf x_k + \Delta \mathbf x \leq \mathbf u.

The steplength :math:`h_k` is determined using a line-search filter developed by Wächter and Biegler. See

    Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for
    large-scale nonlinear programming. Mathematical programming, 106(1), 25-57.


.. automodule:: cgn
   :members:
   :imported-members:
   :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
