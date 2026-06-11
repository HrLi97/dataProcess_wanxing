from .logger import __origin_print__
import builtins


class BuildInLogger:
    """
    Automatically write logs and ensure that the log files are stored in a rolling manner.

    use example:

    with FTSyncLogger("test.log"):
        print("hello")
    """

    def __init__(self):
        self.before_func = __origin_print__

    def __enter__(self):
        self.before_func = builtins.print
        builtins.print = __origin_print__

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        builtins.print = self.before_func
