import torch
import time
from transformers import (
    AutoTokenizer,
    UMT5EncoderModel,
)
from utils_inference.memory import (
    offload_tools,
    FlexibleModuleExtentCudaAsync,
    FlexibleModuleExtent,
    FlexibleMemoryState,
)

FlexibleMemoryState.set_max_memory_usage(1024**3 * 2)

prompt = "this is a bird, only test for utils_inference"
TIMES = 10
device = torch.device("cuda:0")

with torch.no_grad():

    run_dtype = torch.bfloat16
    model_path = "xxxxxxxx"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=run_dtype,
        device=torch.device("cpu"),
    )

    text_inputs = tokenizer(
        [prompt * 100] * 10,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    text_input_ids, mask = text_inputs.input_ids.to(
        device
    ), text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    text_encoder = FlexibleModuleExtentCudaAsync.register_module_extent_by_size(
        text_encoder.eval(),
        calc_device=device,
        min_size=1024**2 * 20,
        max_size=1024**2 * 300,
    )[0]
    # text_encoder = FlexibleModuleExtentCudaAsync.register_module_extent_by_type(
    #     text_encoder.eval(), calc_device=device
    # )[0]
    offload_tools.malloc_pin_memory(text_encoder, contigous=True)
    print(
        f"total regster flexible memory size: {FlexibleMemoryState.total_flexible_register_bytes/1024**3:.2f}GB"
    )

    offload_tools.cuda_module(text_encoder, device)

    torch.cuda.synchronize(device=device)
    start = time.time()

    for i in range(TIMES):
        print("step:", i)
        prompt_embeds = text_encoder(text_input_ids, mask)
    torch.cuda.synchronize()
    end = time.time()
    print(f"avg cost: {(end-start)*1000/TIMES:.2f}ms")

    offload_tools.offload_module(text_encoder)

    print(
        f"finish, peak memory usage: {torch.cuda.max_memory_allocated()/(1024**3):.2f}GB"
    )
    FlexibleMemoryState.release_all()
