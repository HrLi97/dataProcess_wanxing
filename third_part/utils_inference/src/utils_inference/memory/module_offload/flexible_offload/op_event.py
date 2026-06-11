import time
import torch
from concurrent.futures._base import Future
from .event_type import EventType


class OPEvent:
    def __init__(
        self,
        event_type: EventType,
        memory_effect: int = 0,
        start_cuda_event=None,
        end_cuda_event=None,
        cpu_event: Future = None,
        object_id: int = -1,
        object_tag: str = "",
    ):
        """
        description the op event

        if there is a CPU event, it is always necessary to call sync_cpu() to synchronize the CPU event first; otherwise,
        the CUDA event may not take effect (undefined behavior).
        """
        # tags
        self.object_id = object_id
        self.object_tag = object_tag

        self.timestemp = int(time.time()*(10**9))
        self.cpu_event = cpu_event
        self.start_cuda_event = (
            torch.cuda.Event() if start_cuda_event is None else start_cuda_event
        )
        self.end_cuda_event = (
            torch.cuda.Event() if end_cuda_event is None else end_cuda_event
        )
        self.type = event_type
        self.memory_effect = (
            abs(memory_effect)
            if event_type in [EventType.HTOD]
            else -abs(memory_effect)
        )  # only make sense for COPY event, >0 means memory onload, <0 means memory offload

    def sync_cpu(self):
        """
        wait until cpu job finish for current op event. the cuda event can make sense only when the CPU has completed its operation.
        """
        if self.cpu_event is not None:
            self.cpu_event.result()

    def record_start(self, stream: torch.cuda.Stream):
        self.start_cuda_event.record(stream)

    def record_end(self, stream: torch.cuda.Stream):
        self.end_cuda_event.record(stream)

    def __str__(self):
        return f"event:[type={self.type}, id={self.object_id}, tag={self.object_tag}]"
