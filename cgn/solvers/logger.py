"""
Contains class "Logger"
"""

class Logger:
    def __init__(self):
        self.verbose = False

    def log(self, string):
        """
        Logs a given string
        :param string: String
        :return:
            If in verbose mode, the string is displayed to the screen.
            In any case, the string is written to a logfile.
        """
        if self.verbose:
            print(string)

    def newline(self):
        """
        Prints an empty line in the log file and (if verbose) also on screen.
        :return:
        """
        print(" ")