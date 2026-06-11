import torch
from typing import Dict, Optional
from ..module_offload.offload_tools import cuda_module, offload_module


class StorageInfo:
    STORAGE_DTYPE = torch.uint8

    def __init__(self, raw_tensor: torch.Tensor, start: int, length: int):
        self.start = start
        self.length = length
        self.shape = raw_tensor.shape
        self.dtype = raw_tensor.dtype
        self.is_parameter = isinstance(raw_tensor, torch.nn.Parameter)
        self.requires_grad = raw_tensor.requires_grad

    def recover_tensor(self, storage_tensor: torch.Tensor) -> torch.Tensor:
        """
        no data copy, return slice of storage
        """
        if storage_tensor is None:
            return None

        assert storage_tensor.dtype == StorageInfo.STORAGE_DTYPE
        tensor = (
            storage_tensor[self.start : self.start + self.length]
            .view(self.dtype)
            .view(self.shape)
        )

        tensor.requires_grad = self.requires_grad

        if self.is_parameter:
            return torch.nn.Parameter(tensor, requires_grad=self.requires_grad)
        else:
            return tensor

    @staticmethod
    def standardizin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        no data copy, only view change
        """
        return tensor.view((-1,)).view(StorageInfo.STORAGE_DTYPE)


class FederatedStorage:
    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        device: torch.device = None,
        use_pin_memory=True,
        use_contigous_memory=True,
    ):
        """
        read-only once init finish
        """
        self.member_info = {}  # type:Dict[str,StorageInfo]

        self.use_pin_memory = use_pin_memory
        self.use_contigous_memory = use_contigous_memory

        self.storage = None  # type:torch.Tensor
        self.__pin_storage__ = None  # type:torch.Tensor

        self.__federated_tensors__(tensors=tensors, device=device)

    def __federated_tensors__(
        self, tensors: Dict[str, torch.Tensor], device: torch.device = None
    ):
        """
        federated tensors, call repeate will override previous op

        :param tensors: tensor need to federate, must in same device, or give aim device
        :type tensors: Dict[str, torch.Tensor]
        :param device: aim device
        :type device: torch.device
        """
        self.member_info = {}
        count = 0
        standard_tensors = []
        if len(tensors) < 1:
            self.__pin_storage__ = None
            self.storage = None
            return

        for name, tensor in tensors.items():
            if device is not None:
                tensor = tensor.to(device)

            standard_tensor = StorageInfo.standardizin_tensor(tensor)
            self.member_info[name] = StorageInfo(
                tensor, start=count, length=standard_tensor.size(0)
            )
            standard_tensors.append(standard_tensor)
            count += standard_tensor.size(0)
        self.storage = torch.cat(standard_tensors, dim=0)

        if self.use_contigous_memory:
            self.storage = self.storage.contiguous()

        if self.use_pin_memory:
            self.__pin_storage__ = self.storage.cpu().pin_memory()
        else:
            self.__pin_storage__ = None

    def htod(self, device: torch.device = None, non_blocking: bool = True):
        if self.storage is None:
            return

        if self.__pin_storage__ is not None:
            self.storage = self.__pin_storage__.to(
                device=device, non_blocking=non_blocking
            )
        else:
            self.storage = self.storage.to(device=device, non_blocking=non_blocking)

    def dtoh(self):
        if self.storage is None:
            return

        if self.__pin_storage__ is not None:
            self.storage = self.__pin_storage__
        else:
            self.storage = self.storage.cpu()

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = {}
        for name, storage_info in self.member_info.items():
            tensors[name] = storage_info.recover_tensor(self.storage)
        return tensors

    def __len__(self):
        return len(self.member_info)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.member_info[key].recover_tensor(self.storage)

    def __setitem__(self, key, value):
        raise SyntaxError("not allow")

    def __delitem__(self, key):
        raise SyntaxError("not allow")

    @staticmethod
    def register_federate_module(
        module: torch.nn.Module,
        target_device=None,
        skip_flexible=True,
        use_pin_memory=True,
        use_contigous_memory=True,
    ):
        if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
            # ignore Flexible Module
            return

        tensors = {}
        for sub_module_name, sub_module in module.named_modules():
            for name, param in sub_module.named_parameters(recurse=False):
                tensors[f"p_{sub_module_name}.{name}"] = param

            for name, buffer in sub_module.named_buffers(recurse=False):
                tensors[f"b_{sub_module_name}.{name}"] = buffer

        federated_storage = FederatedStorage(
            tensors,
            device=target_device,
            use_pin_memory=use_pin_memory,
            use_contigous_memory=use_contigous_memory,
        )
        setattr(module, "__weights_federated__", federated_storage)

    @staticmethod
    def flush_federate_module(
        module: torch.nn.Module, check_device_cpu: Optional[bool] = None
    ):
        federated_storage = getattr(
            module, "__weights_federated__", None
        )  # type:FederatedStorage
        if federated_storage is None:
            return

        for sub_module_name, sub_module in module.named_modules():
            for name, param in sub_module.named_parameters(recurse=False):
                if check_device_cpu is not None:
                    if (param.device.type == "cpu") != check_device_cpu:
                        print(
                            f"warning federate storage, param not in expected device(cpu={check_device_cpu}), get({param.device.type}) before do any operate"
                        )

                sub_module._parameters[name] = federated_storage[
                    f"p_{sub_module_name}.{name}"
                ]

            for name, buffer in sub_module.named_buffers(recurse=False):
                if check_device_cpu is not None:
                    if (buffer.device.type == "cpu") != check_device_cpu:
                        print(
                            f"warning federate storage, buffer not on expected device(cpu={check_device_cpu}), get({buffer.device.type}) before do any operate"
                        )

                sub_module._buffers[name] = federated_storage[
                    f"b_{sub_module_name}.{name}"
                ]

    @staticmethod
    def cuda_federate_module(
        module: torch.nn.Module, device: torch.device, check_device: bool = True
    ):
        if not hasattr(module, "__weights_federated__"):
            return cuda_module(
                module=module, device=device, contain_sub=True, skip_flexible=False
            )
        else:
            federated_storage = getattr(
                module, "__weights_federated__"
            )  # type:FederatedStorage

            federated_storage.htod(device=device, non_blocking=True)
            FederatedStorage.flush_federate_module(
                module=module, check_device_cpu=True if check_device else None
            )

    @staticmethod
    def offload_federate_module(module: torch.nn.Module, check_device: bool = True):
        if not hasattr(module, "__weights_federated__"):
            return offload_module(module=module, contain_sub=True, skip_flexible=False)
        else:
            federated_storage = getattr(
                module, "__weights_federated__"
            )  # type:FederatedStorage
            federated_storage.dtoh()
            FederatedStorage.flush_federate_module(
                module=module, check_device_cpu=False if check_device else None
            )
