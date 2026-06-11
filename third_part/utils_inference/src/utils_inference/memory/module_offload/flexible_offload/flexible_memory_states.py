import torch
from .flexible_memory_device_state import FlexibleMemoryDeviceState


class FlexibleMemoryState:
    total_flexible_register_bytes = 0
    device_states = {}

    @staticmethod
    def get_device_idx(device: torch.device = None) -> int:
        if device is None:
            return torch.cuda.current_device()
        else:
            assert device.type == "cuda", "only support cuda device"
            return (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )

    @staticmethod
    def add_device_state(device: torch.device = None):
        device_idx = FlexibleMemoryState.get_device_idx(device=device)

        if device_idx not in FlexibleMemoryState.device_states:
            FlexibleMemoryState.device_states[device_idx] = FlexibleMemoryDeviceState(
                device=torch.device(f"cuda:{device_idx}")
            )

    @staticmethod
    def get_device_state(device: torch.device = None) -> FlexibleMemoryDeviceState:
        device_idx = FlexibleMemoryState.get_device_idx(device=device)

        if device_idx not in FlexibleMemoryState.device_states:
            FlexibleMemoryState.add_device_state(device)

        return FlexibleMemoryState.device_states.get(device_idx, None)

    @staticmethod
    def add_flexible_register_bytes(bytes_size: int, device: torch.device = None):
        FlexibleMemoryState.get_device_state(
            device=device
        ).total_flexible_register_bytes += bytes_size
        FlexibleMemoryState.total_flexible_register_bytes += bytes_size

    @staticmethod
    def remove_flexible_register_bytes(bytes_size: int, device: torch.device = None):
        FlexibleMemoryState.get_device_state(
            device=device
        ).total_flexible_register_bytes -= bytes_size
        FlexibleMemoryState.total_flexible_register_bytes -= bytes_size

    @staticmethod
    def reset_flexible_register_bytes(bytes_size: int, device: torch.device = None):
        device_state = FlexibleMemoryState.get_device_state(device=device)
        FlexibleMemoryState.total_flexible_register_bytes -= (
            device_state.total_flexible_register_bytes
        )
        device_state.total_flexible_register_bytes = 0

    # @staticmethod
    # def set_min_distance(min_distance=0, device: torch.device = None):
    #     device_state = FlexibleMemoryState.get_device_state(device=device)
    #     device_state.min_distance = min_distance

    @staticmethod
    def set_max_memory_usage(
        max_memory_usage=5 * (1024**3), device: torch.device = None
    ):
        device_state = FlexibleMemoryState.get_device_state(device=device)
        device_state.max_memory_usage = max_memory_usage

    @staticmethod
    def release(wait: bool = True, device: torch.device = None):
        device_state = FlexibleMemoryState.get_device_state(device=device)
        device_state.release(wait=wait)

    @staticmethod
    def release_all(wait: bool = True):
        for _, device_state in FlexibleMemoryState.device_states.items():
            device_state.release(wait=wait)


FlexibleMemoryState.add_device_state()
