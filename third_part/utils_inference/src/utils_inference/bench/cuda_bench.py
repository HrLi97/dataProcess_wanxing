from concurrent.futures import ThreadPoolExecutor
import time
import torch
from copy import deepcopy
from typing import TypeVar, ParamSpec, Optional, Callable
from functools import wraps

time_bencher = None


def print_cost(
    func_name: str,
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
    cpu_time: float = None,
):
    end_event.synchronize()
    cuda_cost = start_event.elapsed_time(end_event) / 1000.0
    if cpu_time is None:
        print(f"<@{func_name}>: [CUDA]{time.time()}s")
    else:
        print(f"<@{func_name}>: [CPU]{cpu_time:.3f}s [CUDA]{cuda_cost:.3f}s")


def test_time_cuda(enable=True, contain_cpu=True, contain_cuda=True):
    """
    test cpu and cuda time cost, use __debug_test_time__=False to disable current function test during runtime

    :param enable: enable test or not
    """
    T = TypeVar("T")
    P = ParamSpec("P")

    def wrapper0(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(
            *args,
            __debug_test_time__=None,
            **kwargs,
        ) -> T:
            _enable = enable if __debug_test_time__ is None else __debug_test_time__

            if _enable:
                if contain_cpu and not contain_cuda:
                    cpu_start = time.time()
                    output = func(*args, **kwargs)
                    cpu_end = time.time()
                    print(f"<@{func.__qualname__}>: {cpu_end-cpu_start:.3f}s")
                else:
                    global time_bencher
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    cpu_start = time.time()
                    output = func(*args, **kwargs)
                    cpu_end = time.time()
                    end_event.record()

                    if time_bencher is None:
                        time_bencher = ThreadPoolExecutor(
                            thread_name_prefix="cuda_time_bench", max_workers=8
                        )

                    time_bencher.submit(
                        print_cost,
                        func.__qualname__,
                        start_event,
                        end_event,
                        (cpu_end - cpu_start) if contain_cpu else None,
                    )
                return output
            else:
                return func(*args, **kwargs)

        return wrapper

    return wrapper0
