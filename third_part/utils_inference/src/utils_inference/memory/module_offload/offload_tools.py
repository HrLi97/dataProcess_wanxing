import torch
import torch.nn as nn
import gc
from ...bench.time_bench import test_time
from ...common.global_values import GlobalValues
import math
from typing import Dict
from .flexible_offload.flexible_module_extent_base import FlexibleModuleExtentBase
import os


def slice_run(
    func,
    input: torch.Tensor,
    output: torch.Tensor,
    in_dim=0,
    batch_num=2,
    out_dim=None,
):
    if out_dim is None:
        out_dim = in_dim

    in_shape = input.shape[in_dim]
    out_shape = output.shape[out_dim]

    batch_num = min(batch_num, in_shape, out_shape)
    in_stride = math.ceil(in_shape / batch_num)

    if in_shape >= out_shape:
        assert in_shape % out_shape == 0, "bad shape"
        radio = in_shape // out_shape

        out_stride = in_stride // radio
    else:
        assert out_shape % in_shape == 0, "bad shape"
        radio = out_shape // in_shape

        out_stride = in_stride * radio

    for i in range(batch_num):
        start_in = i * in_stride
        end_in = min((i + 1) * in_stride, in_shape)

        start_out = i * out_stride
        end_out = min((i + 1) * out_stride, out_shape)

        if end_in <= start_in or end_out <= start_out:
            break
        out_mem = output.narrow(
            dim=out_dim, start=start_out, length=(end_out - start_out)
        )
        input_mem = input.narrow(dim=in_dim, start=start_in, length=(end_in - start_in))

        out_mem[...] = func(input_mem)


@test_time(enable=GlobalValues.ENABLE_PER)
def flush_memory(sync=True):
    gc.collect()
    if sync:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


@test_time(enable=GlobalValues.ENABLE_PER)
def malloc_pin_memory(
    module: torch.nn.Module,
    ref_device: torch.device = None,
    skip_flexible: bool = True,
    contigous: bool = True,
):
    """
    malloc pin memory for mudule, this can speedup cuda_module and offload_module, but need more host memory.

    :param module: mudule to register pin memory, will modify in-place
    :type module: torch.nn.Module
    :param ref_device: device that will used for compute
    :type ref_device: torch.device
    :param skip_flexible: skip modules with flexible-extent
    :type skip_flexible: bool
    :param contigous: require memory to be contigous or not. True may get faster, but may not be able to handle fragmented memory.
    :type contigous: bool
    """
    return __malloc_pin_memory__(
        module, ref_device, skip_flexible=skip_flexible, contigous=contigous
    )


@test_time(enable=GlobalValues.ENABLE_PER)
def offload_module(
    module: torch.nn.Module,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
):
    """
    module weights DTOH, if module is not registered pin_memory, it will be equal as module=module.cpu() for each graph

    :param module: mudule to set weights DTOH, will modify in-place
    :type module: torch.nn.Module
    :param contain_sub: weights HTOD contain sub modules or not
    :type contain_sub: bool
    :param skip_flexible: skip modules with flexible-extent
    :type skip_flexible: bool
    """
    return __offload_module__(
        module,
        contain_sub=contain_sub,
        skip_flexible=skip_flexible,
        check_device=check_device,
        pre_vious_name=module.__class__.__name__,
    )


@test_time(enable=GlobalValues.ENABLE_PER)
def cuda_module(
    module: torch.nn.Module,
    device: torch.device,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
):
    """
    module weights HTOD, if module is not registered pin_memory, it will be equal as module=module.to(device) for each graph

    :param module: mudule to set weights HTOD, will modify in-place
    :type module: torch.nn.Module
    :param device: device that move to
    :type device: torch.device
    :param contain_sub: weights HTOD contain sub modules or not
    :type contain_sub: bool
    :param skip_flexible: skip modules with flexible-extent
    :type skip_flexible: bool
    """

    return __cuda_module__(
        module,
        device,
        contain_sub=contain_sub,
        skip_flexible=skip_flexible,
        check_device=check_device,
        pre_vious_name=module.__class__.__name__,
    )


def __malloc_pin_memory__(
    module: torch.nn.Module,
    ref_device: torch.device = None,
    skip_flexible: bool = True,
    contigous: bool = True,
):
    """
    malloc pin memory for mudule, this can speedup cuda_module and offload_module, but need more host memory.

    :param module: mudule to register pin memory, will modify in-place
    :type module: torch.nn.Module
    :param ref_device: device that will used for compute
    :type ref_device: torch.device
    :param skip_flexible: skip modules with flexible-extent
    :type skip_flexible: bool
    :param contigous: require memory to be contigous or not. True may get faster, but may not be able to handle fragmented memory.
    :type contigous: bool
    """
    assert module is not None, "module is None"

    if isinstance(module, FlexibleModuleExtentBase):
        return

    if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
        # ignore Flexible Module
        return

    if ref_device is not None:
        for name, module in module.named_modules():
            if hasattr(module, "is_flexible") and module.is_flexible():
                continue
            module.to(ref_device)

    # malloc for parameters
    pin_param = getattr(module, "pin_param", {})
    for name, param in module.named_parameters(recurse=False):
        if name in pin_param:
            continue
        else:
            if contigous:
                param = param.contiguous()
            pin_param[name] = nn.Parameter(
                param.cpu().pin_memory(), requires_grad=param.requires_grad
            )
            module._parameters[name] = pin_param[name]

    setattr(module, "pin_param", pin_param)

    # malloc for buffers
    pin_buffer = getattr(module, "pin_buffer", {})
    for name, buffer in module.named_buffers(recurse=False):
        if name in pin_buffer:
            continue
        else:
            if contigous:
                buffer = buffer.contiguous()
            pin_buffer[name] = buffer.cpu().pin_memory()
            module._buffers[name] = pin_buffer[name]
    setattr(module, "pin_buffer", pin_buffer)

    for sub_module in module.children():
        __malloc_pin_memory__(
            sub_module, ref_device=ref_device, skip_flexible=skip_flexible
        )


def __offload_module__(
    module: torch.nn.Module,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
    pre_vious_name="",
):
    """
    module weights DTOH, if module is not registered pin_memory, it will be equal as module=module.cpu() for each graph

    :param module: mudule to set weights DTOH, will modify in-place
    :type module: torch.nn.Module
    :param contain_sub: weights HTOD contain sub modules or not
    :type contain_sub: bool
    :param skip_flexible: skip modules with flexible-extent
    :type skip_flexible: bool
    """
    assert module is not None, "module is None"

    if isinstance(module, FlexibleModuleExtentBase):
        return

    if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
        # ignore Flexible Module
        return

    DEBUG = eval(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")) != 0
    # copy DtoH
    pin_param = getattr(module, "pin_param", {})  # type: Dict[str,torch.nn.Parameter]
    for name, param in module.named_parameters(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if len(pre_vious_name) > 0 else name
        if check_device:
            if param.device.type == "cpu":
                print(
                    f"warning offload-tools, param({long_name}) not in expected device(cpu=False), get({param.device.type}, id={param.data_ptr()}) before do any operation"
                )

        if DEBUG:
            print(f"offload-tools offload param {long_name}")

        if name not in pin_param:
            # module.register_parameter(name, param.cpu())
            module._parameters[name] = param.cpu()
        else:
            # module.register_parameter(name, pin_param[name])
            module._parameters[name] = pin_param[name]

    pin_buffer = getattr(module, "pin_buffer", {})  # type: Dict[str,torch.Tensor]
    for name, buffer in module.named_buffers(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if len(pre_vious_name) > 0 else name
        if check_device:
            if buffer.device.type == "cpu":
                long_name = (
                    f"{pre_vious_name}.{name}" if len(pre_vious_name) > 0 else name
                )
                print(
                    f"warning offload-tools, buffer({long_name}) not in expected device(cpu=False), get({buffer.device.type}) before do any operation"
                )
        if DEBUG:
            print(f"offload-tools offload buffer {long_name}")

        if name not in pin_buffer:
            # module.register_buffer(name, buffer.cpu())
            module._buffers[name] = buffer.cpu()
        else:
            # module.register_buffer(name, pin_buffer[name])
            module._buffers[name] = pin_buffer[name]

    if contain_sub:
        for name, sub_module in module.named_children():
            if len(name) < 1:
                continue
            __offload_module__(
                sub_module,
                contain_sub=contain_sub,
                skip_flexible=skip_flexible,
                check_device=check_device,
                pre_vious_name=f"{pre_vious_name}.{name}",
            )


def __cuda_module__(
    module: torch.nn.Module,
    device: torch.device,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
    pre_vious_name="",
):
    """
    module weights HTOD, if module is not registered pin_memory, it will be equal as module=module.to(device) for each graph

    :param module: mudule to set weights HTOD, will modify in-place
    :type module: torch.nn.Module
    :param device: device that move to
    :type device: torch.device
    :param contain_sub: weights HTOD contain sub modules or not
    :type contain_sub: bool
    :param skip_flexible: skip modules with flexible-extent
    :type skip_flexible: bool
    """
    assert module is not None, "module is None"

    if isinstance(module, FlexibleModuleExtentBase):
        return

    if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
        return

    DEBUG = eval(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")) != 0

    # copy HtoD
    pin_param = getattr(module, "pin_param", {})  # type: Dict[str,torch.nn.Parameter]
    for name, param in module.named_parameters(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if len(pre_vious_name) > 0 else name
        if check_device:
            if param.device.type != "cpu":
                print(
                    f"warning offload-tools, param({long_name}) not in expected device(cpu=True), get({param.device.type}, id={param.data_ptr()}) before do any operation"
                )

        if DEBUG:
            print(f"offload-tools onload param {long_name}")

        if name not in pin_param:
            module._parameters[name] = param.to(device=device, non_blocking=True)
        else:
            module._parameters[name] = pin_param[name].to(
                device=device, non_blocking=True
            )

    pin_buffer = getattr(module, "pin_buffer", {})  # type: Dict[str,torch.Tensor]
    for name, buffer in module.named_buffers(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if len(pre_vious_name) > 0 else name
        if check_device:
            if buffer.device.type != "cpu":
                print(
                    f"warning offload-tools, buffer({long_name}) not in expected device(cpu=True), get({buffer.device.type}) before do any operation"
                )

        if DEBUG:
            print(f"offload-tools onload buffer {long_name}")

        if name not in pin_buffer:
            module._buffers[name] = buffer.to(device=device, non_blocking=True)
        else:
            module._buffers[name] = pin_buffer[name].to(
                device=device, non_blocking=True
            )

    if contain_sub:
        for name, sub_module in module.named_children():
            if len(name) < 1:
                continue
            __cuda_module__(
                sub_module,
                device=device,
                contain_sub=contain_sub,
                skip_flexible=skip_flexible,
                check_device=check_device,
                pre_vious_name=f"{pre_vious_name}.{name}",
            )
