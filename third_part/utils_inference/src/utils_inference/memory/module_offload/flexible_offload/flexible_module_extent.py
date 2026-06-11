import torch
from torch import nn
from typing import Tuple, Optional, Any
import weakref
from .flexible_memory_states import FlexibleMemoryState
from ...profile.profiler import _record_function_
from .. import offload_tools
from ...storage.federated_storage import FederatedStorage
from .flexible_module_extent_base import (
    FlexibleModuleExtentBase,
    FlexibleSupportStagesEnum,
    FlexibleMemoryTypeEnum,
)
import os


class FlexibleModuleExtent(FlexibleModuleExtentBase):
    """
    Base model for flexible module extent.

    load the weights before running, and offload them after running, only use the current stream.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        calc_device: torch.device,
        use_weak_ref=False,
        init_offload=True,
        memory_contigous=True,
        storage_type: FlexibleMemoryTypeEnum = FlexibleMemoryTypeEnum.FederatedPinMemory,
    ):
        super().__init__()
        self.__use_weak_ref__ = use_weak_ref
        self.__module_ref__ = weakref.ref(module) if self.__use_weak_ref__ else module
        setattr(module, "__forward_record__", getattr(module, "forward"))

        self.__storage_type__ = storage_type

        self.__estimated_memory_requirement__ = (
            FlexibleModuleExtent.get_module_memory_require(
                self.module_ref, contain_grad=False, contain_sub=True
            )
        )  # type:int # count of Bytes

        # define device
        self.set_runtime_device(calc_device)
        self.__calc_device__: torch.device
        self.__current_stream__: torch.cuda.Stream

        # register to FlexibleMemoryState for statistic
        FlexibleMemoryState.add_flexible_register_bytes(
            self.estimated_memory_requirement(), device=self.__calc_device__
        )

        # register memory
        if self.__storage_type__ == FlexibleMemoryTypeEnum.FederatedPinMemory:
            self.register_federated_memory(memory_contigous=memory_contigous)
        elif self.__storage_type__ == FlexibleMemoryTypeEnum.PinMemory:
            self.register_pin_memory(memory_contigous=memory_contigous)
        else:
            self.register_pageable_memory(memory_contigous=memory_contigous)

        if init_offload:
            self.__weight_offload_no_record__(check_device=False)

    @property
    def module_ref(self) -> Optional[nn.Module]:
        assert self.__module_ref__ is not None
        module = self.__module_ref__
        if self.__use_weak_ref__:
            module = module()

        assert isinstance(module, torch.nn.Module)
        return module

    def set_runtime_device(self, device: torch.device):
        self.__calc_device__ = device
        self.__current_stream__ = torch.cuda.current_stream(self.__calc_device__)

    def register_pin_memory(self, memory_contigous=True):
        self.__storage_type__ = FlexibleMemoryTypeEnum.PinMemory
        offload_tools.__malloc_pin_memory__(
            self.module_ref,
            ref_device=None,
            skip_flexible=False,
            contigous=memory_contigous,
        )

    def register_pageable_memory(self, memory_contigous=True):
        self.__storage_type__ = FlexibleMemoryTypeEnum.PageableMemory

    def register_federated_memory(self, memory_contigous=True):
        self.__storage_type__ = FlexibleMemoryTypeEnum.FederatedPinMemory
        FederatedStorage.register_federate_module(
            module=self.module_ref,
            target_device=None,
            skip_flexible=False,
            use_pin_memory=True,
            use_contigous_memory=memory_contigous,
        )

    def estimated_memory_requirement(self) -> int:
        """
        Returning to the estimated memory requirements, this might not be entirely accurate.
        """
        if self.__estimated_memory_requirement__ is None:
            print(
                f"warning: FlexibleBlockOperator need to give __estimated_memory_requirement__ value, None is not allowed."
            )
            return 0
        return max(0, self.__estimated_memory_requirement__)

    def support_train(self) -> FlexibleSupportStagesEnum:
        """
        Only when the backward propagation calculation has been implemented is it allowed to return True.
        """
        return FlexibleSupportStagesEnum.OnlyFunctionWithoutMemOpt

    def support_inference(self) -> FlexibleSupportStagesEnum:
        """
        Only when the backward propagation calculation has been implemented is it allowed to return True.
        """
        return FlexibleSupportStagesEnum.OnlyFunctionWithoutMemOpt

    def weight_onload(self, label=""):
        label = f"weight_onload_{self.estimated_memory_requirement()/(1024**2):.1f}MB_{self.module_ref.__class__.__name__}_{label}"
        if eval(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")) != 0:
            print(label)
        with _record_function_(label):
            self.__weight_onload_no_record__()

    def inference(self, label="", *argc, **kwargs) -> Any:
        label = f"forward_{self.estimated_memory_requirement()/(1024**2):.1f}MB_{self.module_ref.__class__.__name__}_{label}"
        if eval(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")) != 0:
            print(label)
        with _record_function_(label):
            output = self.__inference_no_record__(*argc, **kwargs)
        return output

    def weight_offload(self, label=""):
        label = f"weight_offload_{self.estimated_memory_requirement()/(1024**2):.1f}MB_{self.module_ref.__class__.__name__}_{label}"
        if eval(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")) != 0:
            print(label)
        with _record_function_(label):
            self.__weight_offload_no_record__()

    def __weight_offload_no_record__(self, check_device: bool = True):
        if self.__storage_type__ == FlexibleMemoryTypeEnum.FederatedPinMemory:
            FederatedStorage.offload_federate_module(
                self.module_ref, check_device=check_device
            )
        else:
            offload_tools.__offload_module__(
                module=self.module_ref,
                contain_sub=True,
                skip_flexible=False,
                check_device=check_device,
            )

    def __weight_onload_no_record__(self, check_device: bool = True):
        if self.__storage_type__ == FlexibleMemoryTypeEnum.FederatedPinMemory:
            FederatedStorage.cuda_federate_module(
                self.module_ref, device=self.__calc_device__, check_device=check_device
            )
        else:
            offload_tools.__cuda_module__(
                module=self.module_ref,
                device=self.__calc_device__,
                contain_sub=True,
                skip_flexible=False,
                check_device=check_device,
            )

    def __inference_no_record__(self, *argc, **kwargs) -> Any:
        return self.module_ref.__forward_record__(*argc, **kwargs)

    def forward(self, *argc, **kwargs) -> Any:
        self.__current_stream__ = torch.cuda.current_stream(self.__calc_device__)
        out = super().forward(*argc, **kwargs)
        return out

    @classmethod
    def clean_module_flexible_extent(
        cls,
        module: nn.Module,
    ) -> nn.Module:
        """
        cleanup all exist extent in module

        :param module: module to be cleanup all exist extent in which
        :type module: nn.Module
        :return: module that be cleanup
        :rtype: nn.Module
        """
        for _, submodule in module.named_modules():
            if hasattr(submodule, "flexible_extent") and hasattr(
                submodule, "__forward_record__"
            ):
                delattr(submodule, "flexible_extent")
                delattr(submodule, "is_flexible")
                setattr(
                    submodule,
                    "forward",
                    getattr(submodule, "__forward_record__"),
                )
                delattr(submodule, "__forward_record__")
        return module

    @classmethod
    def offload_all_module_with_flexible_extent(
        cls, module: nn.Module, check_device: bool = True
    ):
        """
        offload current module ant its children if there is flexible extent in which

        :param module: module to offload
        :type module: nn.Module
        :param check_device: check previous device or not
        :type check_device: bool
        """
        for _, sub_module in module.named_modules():
            flexible_extent = getattr(
                sub_module, "flexible_extent", None
            )  # type: FlexibleModuleExtent
            if flexible_extent is not None:
                flexible_extent.__weight_offload_no_record__(check_device=check_device)

    @classmethod
    def onload_all_module_with_flexible_extent(
        cls, module: nn.Module, check_device: bool = True
    ):
        """
        onload current module ant its children if there is flexible extent in which

        :param module: module to offload
        :type module: nn.Module
        :param check_device: check previous device or not
        :type check_device: bool
        """

        for _, sub_module in module.named_modules():
            flexible_extent = getattr(
                sub_module, "flexible_extent", None
            )  # type: FlexibleModuleExtent
            if flexible_extent is not None:
                flexible_extent.__weight_onload_no_record__(check_device=check_device)

    @classmethod
    def register_module_flexible_extent(
        cls,
        module: nn.Module,
        calc_device: torch.device,
        use_weak_ref: bool = True,
        init_offload: bool = True,
        memory_contigous: bool = True,
        storage_type: FlexibleMemoryTypeEnum = FlexibleMemoryTypeEnum.FederatedPinMemory,
        *argc,
        **kwargs,
    ) -> Tuple[nn.Module, bool]:
        """
        add extent to one module, this will auto remove all exist extents in module to avoid nested-extent

        :param module: module to register
        :type module: nn.Module
        :param calc_device: device to real compute
        :type calc_device: torch.device
        :param use_weak_ref: set True for this function
        :type use_weak_ref: bool
        :param init_offload: remove flexible weights to CPU after init, False means remain it still. In any case, the weights of sub-graphs that have not been registered as "flexible" will not affect the motion.
        :type init_offload: bool
        :param memory_contigous: if use pin-memory to speedup, require memory to be contigous or not. True may get faster, but may not be able to handle fragmented memory.
        :type memory_contigous: bool
        :param storage_type: what kind of memory used to offload memory.
        :type storage_type: FlexibleMemoryTypeEnum
        :param argc: other args than class support
        :param kwargs: other args than class support
        :return:
            Module: A module that flexible-extent might be registered. This does not guarantee that the module will be directly registered, it could also be a part of its subgraph.

            bool: True means at least one extent is registered into. False will return Module that give into directly
        :rtype: Tuple[Module, bool]
        """
        if isinstance(module, (nn.ModuleList,)):
            return module, False

        # clean extent in module
        cls.clean_module_flexible_extent(module=module)

        flexible_extent = cls(
            module=module,
            calc_device=calc_device,
            use_weak_ref=use_weak_ref,
            init_offload=init_offload,
            memory_contigous=memory_contigous,
            storage_type=storage_type,
            *argc,
            **kwargs,
        )

        setattr(module, "flexible_extent", flexible_extent)
        setattr(module, "is_flexible", flexible_extent.is_flexible)
        setattr(module, "forward", flexible_extent.forward)
        return module, True

    @classmethod
    def register_module_extent_by_size(
        cls,
        module: nn.Module,
        calc_device: torch.device,
        min_size: int = 50 * 1024 * 1024,
        max_size: int = None,
        use_weak_ref: bool = True,
        init_offload: bool = True,
        memory_contigous: bool = True,
        storage_type: FlexibleMemoryTypeEnum = FlexibleMemoryTypeEnum.FederatedPinMemory,
        *argc,
        **kwargs,
    ) -> Tuple[nn.Module, bool]:
        """
        register flexible extent into mudules, use min~max cuda memory size to control graph-level.

        The integration process will try to reduce the weight-size of the subgraphs as much as possible to maintain a small granularity. always start the fusion from the leaf sub-module

        :param module: module to register
        :type module: nn.Module
        :param calc_device: device to real compute
        :type calc_device: torch.device
        :param min_size: sub-graph weight >=min_size can be added flexible extent, some sub-graph with low-usage weight may skip or fused into father graph
        :type min_size: int
        :param max_size: sub-graph weight <=max_size can be added flexible extent. None means max(2*min_size, int(memory_poll_size/3), min_size)
        :type max_size: int
        :param use_weak_ref: set True for this function
        :type use_weak_ref: bool
        :param init_offload: remove flexible weights to CPU after init, False means remain it still. In any case, the weights of sub-graphs that have not been registered as "flexible" will not affect the motion.
        :type init_offload: bool
        :param memory_contigous: if use pin-memory to speedup, require memory to be contigous or not. True may get faster, but may not be able to handle fragmented memory.
        :type memory_contigous: bool
        :param storage_type: what kind of memory used to offload memory.
        :type storage_type: FlexibleMemoryTypeEnum
        :param argc: other args than class support
        :param kwargs: other args than class support
        :return:
            Module: A module that flexible-extent might be registered. This does not guarantee that the module will be directly registered, it could also be a part of its subgraph.

            bool: True means at least one extent is registered into. False will return Module that give into directly
        :rtype: Tuple[Module, bool]
        """
        if cls.check_flexible(module, contiain_sub=False):
            return module, True

        state = FlexibleMemoryState.get_device_state(device=calc_device)
        if max_size is None:
            max_size = max(min_size * 2, int(state.max_memory_usage / 3), min_size)

        require_size = cls.get_module_memory_require(module)
        if require_size < min_size:
            return module, False

        exist_replace = False
        for name, child_module in module.named_children():
            if name == "":
                continue
            new_module, replace = cls.register_module_extent_by_size(
                child_module,
                calc_device=calc_device,
                min_size=min_size,
                max_size=max_size,
                use_weak_ref=use_weak_ref,
                init_offload=init_offload,
                memory_contigous=memory_contigous,
                storage_type=storage_type,
                *argc,
                **kwargs,
            )
            if replace:
                setattr(module, name, new_module)

            exist_replace = exist_replace or replace

        if exist_replace:
            return module, True
        elif require_size > max_size:
            return module, False
        else:
            return cls.register_module_flexible_extent(
                module,
                calc_device=calc_device,
                use_weak_ref=use_weak_ref,
                init_offload=init_offload,
                memory_contigous=memory_contigous,
                storage_type=storage_type,
                *argc,
                **kwargs,
            )

    @classmethod
    def register_module_extent_by_type(
        cls,
        module: nn.Module,
        calc_device: torch.device,
        types: list = [nn.Linear, nn.Conv2d, nn.Conv3d],
        size: int = 60 * 1024 * 1024,
        use_weak_ref: bool = True,
        init_offload: bool = True,
        memory_contigous: bool = True,
        storage_type: FlexibleMemoryTypeEnum = FlexibleMemoryTypeEnum.FederatedPinMemory,
        *argc,
        **kwargs,
    ) -> Tuple[nn.Module, bool]:
        """
        register flexible extent into mudules, use type and min cuda memory size to select graph-level

        :param module: module to register
        :type module: nn.Module
        :param calc_device: device to real compute
        :type calc_device: torch.device
        :param types: class-types that can be register extent
        :type types: list
        :param size: only >=min_size can be added flexible extent
        :type size: int
        :param use_weak_ref: set True for this function
        :type use_weak_ref: bool
        :param init_offload: remove flexible weights to CPU after init, False means remain it still. In any case, the weights of sub-graphs that have not been registered as "flexible" will not affect the motion.
        :type init_offload: bool
        :param memory_contigous: if use pin-memory to speedup, require memory to be contigous or not. True may get faster, but may not be able to handle fragmented memory.
        :type memory_contigous: bool
        :param storage_type: what kind of memory used to offload memory.
        :type storage_type: FlexibleMemoryTypeEnum
        :param argc: other args than class support
        :param kwargs: other args than class support
        :return:
            Module: A module that flexible-extent might be registered. This does not guarantee that the module will be directly registered, it could also be a part of its subgraph.

            bool: True means at least one extent is registered into. False will return Module that give into directly
        :rtype: Tuple[Module, bool]
        """
        if cls.check_flexible(module, contiain_sub=False):
            return module, True

        if (
            isinstance(module, tuple(types))
            and cls.get_module_memory_require(module) > size
        ):
            return cls.register_module_flexible_extent(
                module,
                calc_device=calc_device,
                use_weak_ref=use_weak_ref,
                init_offload=init_offload,
                memory_contigous=memory_contigous,
                storage_type=storage_type,
                *argc,
                **kwargs,
            )

        exist_replace = False
        for name, child_module in module.named_children():
            if name == "":
                continue
            new_module, replace = cls.register_module_extent_by_type(
                child_module,
                calc_device=calc_device,
                types=types,
                size=size,
                use_weak_ref=use_weak_ref,
                init_offload=init_offload,
                memory_contigous=memory_contigous,
                storage_type=storage_type,
                *argc,
                **kwargs,
            )
            if replace:
                setattr(module, name, new_module)

            exist_replace = exist_replace or replace

        return module, exist_replace
