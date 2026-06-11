from torch.profiler import record_function
from contextlib import nullcontext, contextmanager
import torch
from functools import wraps
from typing import TypeVar, ParamSpec, Optional, Callable, Any, overload
import time

try:
    _nvtx = torch.cuda.nvtx
    _has_nvtx = True
except:
    _nvtx = None
    _has_nvtx = False

ENABLE_PROF_MARKS = True


@contextmanager
def prof_range(name: str):
    """Create a profiler range for the current scope."""
    rf = record_function(name)

    if _has_nvtx:
        _nvtx.range_push(name)
        with rf:
            try:
                yield
            finally:
                _nvtx.range_pop()
    else:
        with rf:
            try:
                yield
            finally:
                pass


def mark(msg: str):
    """Insert a profiler mark for the current scope."""
    if _has_nvtx and ENABLE_PROF_MARKS:
        _nvtx.mark(msg)
    else:
        pass


def _record_function_(name: str, add_unique_tail: bool = False):
    """
    Record function for profiling and easy debugging

    with _record_function_(name=tag):
        output = func(...)
    """
    if add_unique_tail:
        name = f"{name}_{str(int(time.time() * (10**4)))}"

    if ENABLE_PROF_MARKS:
        return prof_range(name)
    else:
        return nullcontext()


def _record_function_wrap_(name: str, task_id: str = None, auto_id: bool = False):
    """
    Record function for profiling and easy debugging

    @_record_function_wrap_(...)
    def func(...):
        pass
    """

    T = TypeVar("T")
    P = ParamSpec("P")

    def wrapper0(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            _task_id, _auto_id = task_id, auto_id
            if _task_id is None and _auto_id:
                _task_id = str(int(time.time() * (10**4)))

            if _task_id is not None and len(_task_id) > 0:
                tag = f"{name}-{_task_id}"
            else:
                tag = f"{name}"
            with _record_function_(name=tag):
                output = func(*args, **kwargs)
            return output

        return wrapper

    return wrapper0
