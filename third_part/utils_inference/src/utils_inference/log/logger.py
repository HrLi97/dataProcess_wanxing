import sys, os, io, builtins
from logging.handlers import RotatingFileHandler
import logging

# from rich.console import Console
# from rich.text import Text

__origin_print__ = builtins.print


def print_str(*args, **kwargs) -> str:
    """
    get print content as str
    """
    buf = io.StringIO()
    kwargs.setdefault("file", buf)
    __origin_print__(*args, **kwargs)
    return buf.getvalue()


class FTSyncLogger:
    """
    Automatically write logs and ensure that the log files are stored in a rolling manner.

    use example:

    with FTSyncLogger("test.log"):
        print("hello")
    """

    def __init__(self, filename=None, prompt: str = "", console=None, skip_print=False):
        self.logfile = None
        self.loghandle = None
        self.prompt = prompt
        self.console = console
        self.skip_print = skip_print
        self.rich_prompt = f"\033[32m{prompt}\033[0m " if len(prompt) > 0 else ""
        if self.console is not None and len(prompt) > 0:
            from rich.text import Text
            from rich.console import Console

            self.console: Console
            self.rich_prompt = Text(self.prompt + " ", style="green")

        if filename is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            self.loghandle = RotatingFileHandler(
                filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            self.loghandle.setFormatter(
                logging.Formatter(
                    "[%(asctime)s %(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

            self.logfile = logging.getLogger(filename)
            self.logfile.propagate = False
            self.logfile.setLevel(logging.INFO)
            self.logfile.addHandler(self.loghandle)

            self.logfile.info(
                "\n\n===================== new log =====================\n"
            )

    def __del__(self):
        if self.logfile is not None:
            self.loghandle.flush()
            self.loghandle.close()
            self.loghandle = None
            self.logfile = None
        self.__exit__()

    def get_print_func(self):
        def print_func(*args, **kwargs):
            string = print_str(*args, **kwargs)
            if len(string) > 0 and string[-1] == "\n":
                string = string[:-1]

            if not self.skip_print:
                if self.console is not None:
                    from rich.text import Text
                    self.console.print(self.rich_prompt + Text(string, style="reset"))
                else:
                    __origin_print__(self.rich_prompt + string)

            if self.logfile is not None:
                self.logfile.info(f"{self.prompt} {string}")

        return print_func

    def __enter__(self):
        builtins.print = self.get_print_func()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        builtins.print = __origin_print__
