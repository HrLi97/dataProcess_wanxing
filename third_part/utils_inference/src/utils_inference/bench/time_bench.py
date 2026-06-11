import time
import time
import inspect
from typing import TypeVar, ParamSpec, Optional, Callable, Any, overload
from copy import deepcopy
from functools import wraps


class TimeStamp:
    def __init__(self):
        self.file = ""
        self.function_name = ""
        self.line = 0
        self.time = None
        self.tip = ""

    def __call__(self, reset=False, tip=None, detail=False, *args, **kwds):
        current_time = time.time()
        stack = inspect.stack()
        caller = stack[1]
        current_file = caller.filename
        current_function_name = caller.function
        current_line = caller.lineno
        if tip is not None:
            self.tip = f"<<{tip}>> "

        if self.time is not None and not reset:
            if not detail:
                print(
                    f"{self.tip if self.tip is not None else ''}{self.function_name}@{self.line}->{current_function_name}@{current_line}: {time.time()-self.time:.3f}s"
                )
            else:
                print(
                    f"{self.tip if self.tip is not None else ''}{self.file}@{self.function_name}: {self.line}->{current_file}@{current_function_name}:{current_line}:{(current_time-self.time):.3f}s"
                )
        self.file = current_file
        self.function_name = current_function_name
        self.line = current_line
        self.time = current_time

    def reset(self, remain_tip=False):
        self.time = None
        if not remain_tip:
            self.tip = None

    def exit(self):
        stack = inspect.stack()
        caller = stack[1]
        print(f"<exit> {caller.filename}@{caller.function}: {caller.lineno}")
        exit(0)


gtime_stamp = TimeStamp()


def test_time(enable=True):
    """
    test cpu time cost, use __debug_test_time__=False to disable current function test during runtime

    :param enable: enable test or not
    """
    T = TypeVar("T")
    P = ParamSpec("P")

    def wrapper0(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(
            *args: P.args,
            __debug_test_time__: Optional[bool] = None,
            **kwargs: P.kwargs,
        ) -> T:
            _enable = enable if __debug_test_time__ is None else __debug_test_time__
            if _enable:
                start = time.time()
                output = func(*args, **kwargs)
                print(f"<@{func.__qualname__}>: {time.time()-start:.3f}s")
                return output
            else:
                return func(*args, **kwargs)

        return wrapper

    return wrapper0
