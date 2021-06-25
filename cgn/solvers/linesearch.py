"""
Contains class "Linesearch".
"""

class Linesearch:
    """
    Abstract base class for line search methods.
    """
    def _check_options(self, options):
        """
        Checks if options is a dict or None
        :param options:
        :return: It returns options. If options was None, it is now an empty dict.
        """
        if options is None:
            options = {}
        else:
            assert isinstance(options, dict), "'linesearchOptions' must be a dict"
        return options

    def _set_options(self, linesearch_options):
        raise NotImplementedError