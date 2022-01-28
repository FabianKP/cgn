
class Solveroptions:
    """
    The options for the constrained Gauss-Newton method.

    :ivar maxiter: The maximum number of iterations. Defaults to 100.
    :ivar tol: The tolerance. The solver stops once the difference between the current (c0) and the last (c1) value of
        the scaled costfunction is less or equal to `tol * scale`, where `scale = max(abs(c0), abs(c1), 1)`.
        Defaults to 1e-10.
    :ivar timeout: Gives the number of minutes after which the solver automatically terminates due to timeout.
        Default timeout is 60 minutes.
    :ivar logfile: Specifies the name of the logfile for the solver output.
    :ivar ctol: The tolerance for the constraint violation. The solver will only try to satisfy the constraint up to
        a tolerance specified by `ctol`. The default value is 1e-15.
    """
    def __init__(self):
        self.maxiter = 100
        self.tol = 1e-10
        self.timeout = 60
        self._verbose = 1
        self.logfile = "cgn.log"
        self.ctol = 1e-10

    @property
    def verbosity(self):
        return self._verbose

    def set_verbosity(self, lvl: int):
        """
        Controls the amount of information displayed during and after the iteration.

        :param lvl:
            The verbosity level. There are currently 2 levels of verbosity:
                - 0: no output
                - 1: output to logfile (specified by :py:attr:`~logfile`.)
                - 2: output to console at running time
                - 3: output both to console and logfile
        """
        if not lvl in [0, 1, 2, 3]:
            print("'lvl' must be 0, 1, 2, or 3 (see also documentation)")
        else:
            self._verbose = lvl
