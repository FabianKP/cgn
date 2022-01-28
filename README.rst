CGN
============

**CGN** stands for "Constrained Gauss-Newton method".
This package contains a **Python** implementation of the Gauss-Newton method for nonlinear least-squares
problems with linear constraints. That is, CGN provides a method for solving constrained optimization
problems of the form

.. image:: ncnlsp.png


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


Usage
========

You can implement a problem of the above form by filling out the template below.

.. literalinclude:: template.py
    :language: python


For a more details, for example on how to configure solver options, see the `tutorial <tutorial>`_ and `documentation <docs>`_.
