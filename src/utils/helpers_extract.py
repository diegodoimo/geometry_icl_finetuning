import sys
import torch
from .dataloader_utils import get_dataloader


def get_target_layers(
    model, n_layer, option="norm1", every=1, world_size=1, finetuned=False
):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    prefix = "module."
    middle = ""
    # ?? accelerate does not cast to bf16 a DDP model yet
    if world_size > 1:
        prefix = "_fsdp_wrapped_module."
        if map_names[option] != "":
            middle = "._fsdp_wrapped_module"
    if finetuned:
        prefix = "base_model.model."

    target_layers = {
        i: f"{prefix}model.layers.{i}{middle}{suffix}" 
        for i in range(0, n_layer, every)
    }

    target_layers[n_layer] = f"{prefix}model.norm"
    target_layers[n_layer + 1] = f"{prefix}lm_head"
    
    if target_layers[n_layer] not in names:
        target_layers = {
            i: f"model.layers.{i}{middle}{suffix}"
            for i in range(0, n_layer, every)
        }
    
    for target_layer in target_layers.values():
        assert target_layer in names, f"{target_layer} not found in model"

    return target_layers


def print_memory_consumed(rank=0):
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    if rank == 0:
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
    sys.stdout.flush()


@torch.no_grad()
def is_memory_enough(model, longest_seq, micro_batch_size, pad_token_id, world_size):
    # just a series of forward passes with the longest sequences
    model = model.eval()
    longest_loader = get_dataloader(
        longest_seq,
        micro_batch_size,
        pad_token_id,
        world_size=world_size,
        shuffle=False,
        num_processes=4,
    )

    for i, batch in enumerate(longest_loader):
        if world_size == 1:
            batch = {key: val.to("cuda") for key, val in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    torch.cuda.empty_cache()
