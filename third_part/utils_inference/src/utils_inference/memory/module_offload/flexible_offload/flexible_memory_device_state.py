import torch

from collections import deque
from typing import Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future
from .op_event import OPEvent
from .event_type import EventType


class FlexibleMemoryDeviceState:
    def __init__(
        self,
        device: torch.device,
        async_worker: int = 8,
        compute_with_default_stream=True,
    ):
        # states
        self.total_flexible_register_bytes = 0
        self.flexible_usage_bytes = 0  # may <0, show memory usage out of queue
        self.flexible_wait_onload = 0  # alwary >=0, show memory require in queue
        self.flexible_wait_offload = 0  # alwary >=0, show memory require in queue

        self.__last_offload_op_event__ = None

        # self.min_distance = 0
        self.max_memory_usage = 5 * (
            1024**3
        )  # 5GB, if one operator cannot be loaded in anyway, this parameter will be temporarily ignored.

        # streams
        self.htod_stream = torch.cuda.Stream(device)
        self.compute_stream = (
            torch.cuda.current_stream(device)
            if compute_with_default_stream
            else torch.cuda.Stream(device)
        )
        self.dtoh_stream = torch.cuda.Stream(device)

        # event_queue
        self.event_queue = deque()  # type:deque[OPEvent]

        # async worker
        self.async_worker = ThreadPoolExecutor(max_workers=async_worker)

    def add_op_event(self, op_event: OPEvent, skip_calc=True):
        if skip_calc and op_event.type == EventType.CALC:
            return

        if op_event.memory_effect > 0:
            self.flexible_wait_onload += abs(op_event.memory_effect)
        else:
            self.flexible_wait_offload += abs(op_event.memory_effect)
        self.event_queue.append(op_event)

    def remove_op_event(self, count=1) -> Tuple[Optional[OPEvent], bool]:
        last_event = None
        for _ in range(count):
            if len(self.event_queue) < 1:
                return last_event, False

            last_event = self.event_queue.popleft()  # type:OPEvent
            if last_event.memory_effect > 0:
                self.flexible_wait_onload -= abs(last_event.memory_effect)
            else:
                self.flexible_wait_offload -= abs(last_event.memory_effect)

            self.flexible_usage_bytes += last_event.memory_effect
        return last_event, True  # return last event

    def remove_op_event_to_timestemp(
        self, timestemp: int
    ) -> Tuple[Optional[OPEvent], bool]:
        """
        remove event until timestemp in last event that be removed or no event in list anymore
        """

        last_event = None
        while True:
            if len(self.event_queue) < 1:
                return last_event, False

            last_event = self.event_queue.popleft()  # type:OPEvent
            if last_event.memory_effect > 0:
                self.flexible_wait_onload -= abs(last_event.memory_effect)
            else:
                self.flexible_wait_offload -= abs(last_event.memory_effect)

            self.flexible_usage_bytes += last_event.memory_effect

            if last_event.timestemp == timestemp:
                return last_event, True  # return last event

    def find_op_rely(
        self, require_memory: int, max_memory: int = -1
    ) -> Optional[OPEvent]:
        """
        Just for calculation, it does not actually affect the data. Return Rely Event

        :param require_memory: cuda memoery need to use
        :type require_memory: int
        :param max_memory: max cuda memoery allow to use for flexible offload
        :type max_memory: int
        :return: op_event need to wait finish, none means no reliance.
        :rtype: OPEvent | None
        """
        max_memory = self.max_memory_usage if max_memory < 0 else max_memory
        usage_memory = self.flexible_usage_bytes
        wait_onload_memory = self.flexible_wait_onload
        wait_offload_memory = self.flexible_wait_offload

        last_op_event = None
        for op_event in self.event_queue:
            if usage_memory + wait_onload_memory + require_memory <= max_memory:
                return last_op_event

            # memory is not enough
            if op_event.memory_effect > 0:
                wait_onload_memory -= abs(op_event.memory_effect)
            else:
                wait_offload_memory -= abs(op_event.memory_effect)
                last_op_event = op_event  # can only rely on op_event in other stream different HTODs

            usage_memory += op_event.memory_effect

        return last_op_event

    def add_async_job(self, func: Callable, *args, **kwargs) -> Future:
        return self.async_worker.submit(func, *args, **kwargs)

    def release(self, wait: bool = True):
        self.async_worker.shutdown(wait=wait)
