import torch
from torch import nn
from .flexible_memory_states import FlexibleMemoryState
from .op_event import OPEvent
from .event_type import EventType
from .flexible_module_extent import (
    FlexibleModuleExtent,
    FlexibleSupportStagesEnum,
    FlexibleMemoryTypeEnum,
)
from typing import Any


class FlexibleModuleExtentCudaAsync(FlexibleModuleExtent):
    """
    no support for train, not support for share_tensor in weights cross modules
    """

    def __init__(
        self,
        module: nn.Module,
        calc_device: torch.device,
        use_weak_ref=False,
        init_offload=True,
        memory_contigous=True,
        storage_type: FlexibleMemoryTypeEnum = FlexibleMemoryTypeEnum.FederatedPinMemory,
    ):
        super().__init__(
            module=module,
            calc_device=calc_device,
            use_weak_ref=use_weak_ref,
            init_offload=init_offload,
            memory_contigous=memory_contigous,
            storage_type=storage_type,
        )

        self.__offload_cpu_job__ = None
        self.register_events()

    def register_events(self):
        # onload
        self.__onload_start_cuda_event__ = torch.cuda.Event()
        self.__onload_end_cuda_event__ = torch.cuda.Event()

        # offload
        self.__offload_start_cuda_event__ = torch.cuda.Event()
        self.__offload_end_cuda_event__ = torch.cuda.Event()

        # inference
        self.__inference_start_cuda_event__ = torch.cuda.Event()
        self.__inference_end_cuda_event__ = torch.cuda.Event()

    def support_train(self) -> FlexibleSupportStagesEnum:
        """
        Only when the backward propagation calculation has been implemented is it allowed to return True.
        """
        return FlexibleSupportStagesEnum.NoneSupport

    def support_inference(self) -> FlexibleSupportStagesEnum:
        """
        Only when the backward propagation calculation has been implemented is it allowed to return True.
        """
        return FlexibleSupportStagesEnum.FullSupport

    def weight_onload(self, label=""):
        """
        weight onload with CPU-Sync and cuda-Async
        """
        state = FlexibleMemoryState.get_device_state(device=self.__calc_device__)
        onload_event = OPEvent(
            event_type=EventType.HTOD,
            memory_effect=self.estimated_memory_requirement(),
            start_cuda_event=self.__onload_start_cuda_event__,
            end_cuda_event=self.__onload_end_cuda_event__,
            object_id=id(self.module_ref),
            object_tag=self.module_ref.__class__.__name__,
        )

        with torch.cuda.stream(state.htod_stream):
            """
            onload weights
            """
            # wait for privious offload, if use pin_memory, offload signal is inference_end (only CPU OP)
            # ensure offload finish
            if self.__offload_cpu_job__ is not None:
                self.__offload_cpu_job__.result()  # wait until CPU ops in weights offload finish
                self.__offload_cpu_job__ = None

            self.__offload_end_cuda_event__.wait()

            # wait until memory enough
            rely_op_event = state.find_op_rely(
                onload_event.memory_effect, max_memory=-1
            )
            if rely_op_event is not None:
                state.remove_op_event_to_timestemp(rely_op_event.timestemp)
                rely_op_event.sync_cpu()
                rely_op_event.end_cuda_event.wait()

            # load memory
            onload_event.record_start(torch.cuda.current_stream(self.__calc_device__))
            super().weight_onload(label=label)
            onload_event.record_end(torch.cuda.current_stream(self.__calc_device__))

        state.add_op_event(onload_event)

    def inference(self, label="", *argc, **kwargs) -> Any:
        """
        inference compute with CPU-Sync and cuda-Async
        """
        state = FlexibleMemoryState.get_device_state(device=self.__calc_device__)
        calc_event = OPEvent(
            event_type=EventType.CALC,
            memory_effect=0,
            start_cuda_event=self.__inference_start_cuda_event__,
            end_cuda_event=self.__inference_end_cuda_event__,
            object_id=id(self.module_ref),
            object_tag=self.module_ref.__class__.__name__,
        )

        with torch.cuda.stream(state.compute_stream):
            """
            compute forward
            """
            torch.cuda.current_stream(self.__calc_device__).wait_stream(
                self.__current_stream__
            )
            self.__onload_end_cuda_event__.wait()

            calc_event.record_start(torch.cuda.current_stream(self.__calc_device__))
            output = super().inference(label, *argc, **kwargs)
            calc_event.record_end(torch.cuda.current_stream(self.__calc_device__))

        state.add_op_event(calc_event)
        return output

    def _weight_offload_sync_(
        self,
        offload_event: OPEvent,
        offload_stream: torch.cuda.Stream,
        label="",
    ) -> bool:
        with torch.cuda.stream(offload_stream):
            """
            offload weights
            """
            self.__inference_end_cuda_event__.wait()
            self.__inference_end_cuda_event__.synchronize()  # sync CPU and CUDA until op compute finish
            # queue.flag
            offload_event.record_start(torch.cuda.current_stream(self.__calc_device__))
            super().weight_offload(label=label)  # may only with CPU op

            offload_event.record_end(torch.cuda.current_stream(self.__calc_device__))
        return True

    def weight_offload(self, label=""):
        """
        offload weights with CPU-Async and cuda-Async
        """
        state = FlexibleMemoryState.get_device_state(device=self.__calc_device__)
        offload_event = OPEvent(
            event_type=EventType.DTOH,
            memory_effect=self.estimated_memory_requirement(),
            start_cuda_event=self.__offload_start_cuda_event__,
            end_cuda_event=self.__offload_end_cuda_event__,
            object_id=id(self.module_ref),
            object_tag=self.module_ref.__class__.__name__,
        )
        self.__offload_cpu_job__ = state.add_async_job(
            self._weight_offload_sync_, offload_event, state.dtoh_stream, label
        )
        offload_event.cpu_event = self.__offload_cpu_job__
        state.add_op_event(offload_event)
