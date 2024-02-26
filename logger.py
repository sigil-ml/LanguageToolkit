import logging
import sys
import os
import traceback
import datetime
from rich.console import Console
from rich.table import Table
from contextlib import contextmanager
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.jupyter import print


# install(show_locals=True)
console = Console()
error_console = Console(stderr=True, style="bold red")


class CustomFormatter(logging.Formatter):
    """Logging colored formatter"""

    green = "\u001b[32m"
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    table = Table(title="Logs", show_lines=True)
    console = Console()

    def __init__(self):
        super().__init__()

        def msg_factory(color):
            return f"| {self.green}%(asctime)s{self.reset} | {{%(filename)s | %(funcName)s | %(lineno)d}} | %(levelname)s |"

        self.FORMATS = {
            logging.DEBUG: msg_factory(self.blue),
            logging.INFO: msg_factory(self.blue),
            logging.WARNING: msg_factory(self.blue),
            logging.ERROR: msg_factory(self.blue),
            logging.CRITICAL: msg_factory(self.blue),
        }
        columns = ["Time", "File", "Function", "Line", "Level", "Message"]

        for column in columns:
            self.table.add_column(column)

    def format(self, record):
        # row = [
        #     str(datetime.date.today()),
        #     "<" + record.filename + ">",
        #     record.funcName,
        #     str(record.lineno),
        #     record.levelname,
        #     record.msg,
        # ]
        # self.table.add_row(*row)
        # self.console.print(self.table)
        # print(self.table)
        # log_fmt = self.FORMATS.get(record.levelno)
        # formatter = logging.Formatter(log_fmt)
        console.log(record.msg)
        return ""


# logging.basicConfig(
#     level=logging.CRITICAL,
#     format="[%(asctime)s]{%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
#     stream=sys.stdout,
# )

# Create custom logger logging all five levels
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Define format for logs

# Create stdout handler for logging to the console (logs all five levels)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(CustomFormatter())
logger.addHandler(stdout_handler)


def exception_hook(exc_type, exc_value, exc_traceback):
    exc_info = sys.exc_info()
    message = traceback.format_exception(exc_value)
    error_console.log(f"Uncaught exception: {''.join(message)}")


sys.excepthook = exception_hook
