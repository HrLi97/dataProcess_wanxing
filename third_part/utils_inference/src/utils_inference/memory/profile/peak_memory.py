import torch
from contextlib import contextmanager
import time

__memory_stack__ = []


def get_current_time_flag() -> int:
    return time.time_ns()


def record_start(start_memory: float = 0) -> int:
    global __memory_stack__
    __memory_stack__.append(start_memory)
    return len(__memory_stack__) - 1


def record_memory(index_flag: int, peak_memory: float) -> float:
    global __memory_stack__
    peak = max(peak_memory, max(__memory_stack__[index_flag:]))
    if index_flag > 0:
        # still has privious node, need to record peak
        __memory_stack__ = __memory_stack__[:index_flag] + [peak]
    else:
        # no privious node, no record
        __memory_stack__ = []
    return peak


def peak_memory_wrap(sync=False, device: torch.device = None, enable: bool = True):
    """
    unsafe for multi-threads
    """

    def _wrapper_(func):
        if not enable:
            return func

        def wrapper(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats(device=device)
            if sync:
                torch.cuda.synchronize(device=device)
            start_memory = torch.cuda.memory_allocated(device) / (1024**3)
            index_flag = record_start(start_memory)
            result = func(*args, **kwargs)
            if sync:
                torch.cuda.synchronize(device=device)
            end_peak_memory = record_memory(
                index_flag, torch.cuda.max_memory_allocated(device) / (1024**3)
            )
            end_memory = torch.cuda.memory_allocated(device) / (1024**3)
            print(
                f"<@{func.__qualname__}> peak_memory: from {start_memory:.3f}GB to {end_memory:.3f}GB, peak {end_peak_memory:.3f}GB, peak memory add {end_peak_memory-start_memory:.3f}GB"
            )

            return result

        return wrapper

    return _wrapper_


@contextmanager
def peak_memory_context(
    label: str, sync=False, device: torch.device = None, enable: bool = True
):
    """
    unsafe for multi-threads
    """
    if enable:
        torch.cuda.reset_peak_memory_stats(device=device)
        if sync:
            torch.cuda.synchronize(device=device)
        start_memory = torch.cuda.memory_allocated(device) / (1024**3)
        index_flag = record_start(start_memory)

        yield

        if sync:
            torch.cuda.synchronize(device=device)
        end_peak_memory = record_memory(
            index_flag, torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        end_memory = torch.cuda.memory_allocated(device) / (1024**3)
        print(
            f"<{label}> peak_memory: from {start_memory:.3f}GB to {end_memory:.3f}GB, peak {end_peak_memory:.3f}GB, peak memory add {end_peak_memory-start_memory:.3f}GB"
        )
    else:
        yield


if __name__ == "__main__":
    device = torch.device("cuda:0")
    _ = torch.empty(0, device=device)  # init pytorch

    @peak_memory_wrap(device=device)
    def func():
        with peak_memory_context("define a", device=device):
            a = torch.randn(10000, 10000, device=device)
        with peak_memory_context("define b", device=device):
            b = torch.randn(10000, 10000, device=device)
        with peak_memory_context("mul c", device=device):
            c = a @ b
            del a, b
        with peak_memory_context("mul_ c", device=device):
            c.mul_(2)

    with peak_memory_context("test_context", device=device):
        func()
