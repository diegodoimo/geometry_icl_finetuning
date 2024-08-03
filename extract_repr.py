#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import os
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger

import transformers
import sys
from utils.helpers_extract import get_target_layers
from utils.model_utils import get_model
from utils.dataloader_utils import get_dataloader
from utils.dataset_utils import mmlu_dataset
from utils.scienceqa import scienceqa_dataset
from utils.mmlu_pro_race import mmlu_pro_race
from utils.tokenizer_utils import get_tokenizer
from extraction.compute_distances import estract_representations
import torch
import os
from utils.helpers_extract import print_memory_consumed, is_memory_enough

from functools import partial
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)

from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mmlu",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--out_filename", type=str, default="", help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Print every logging_steps samples processed.",
    )
    parser.add_argument(
        "--use_last_token",
        action="store_true",
        help="If passed, ID will be measured on the last token represenation.",
    )
    parser.add_argument(
        "--save_distances",
        action="store_true",
        help="If passed, the distance matrices will be saved",
    )
    parser.add_argument(
        "--save_repr",
        action="store_true",
        help="If passed, the distance matrices will be saved",
    )
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        help="If passed, duplicate datapoints will be removed",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="norm1",
        help="The name of the layer to analyze.",
    )
    parser.add_argument(
        "--maxk",
        type=int,
        default=50,
        help="max nn order of the stored distence matrices",
    )
    parser.add_argument(
        "--layer_interval",
        type=int,
        default=1,
        help="Extract 1 layer every 'layer interval'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="model_name.",
    )
    parser.add_argument(
        "--num_few_shots",
        type=int,
        default=0,
        help="number_few_shots",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--finetuned_mode",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--finetuned_epochs",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt_epoch", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--dev_index", type=int, default=None)
    parser.add_argument("--sample_questions", action="store_true")
    parser.add_argument("--random_order", action="store_true")
    parser.add_argument("--few_shot_topics", action="store_true")
    parser.add_argument("--prompt_mmlu", action="store_true")
    args = parser.parse_args()
    return args


def lambda_fn(module: torch.nn.Module):
    if isinstance(module, LlamaDecoderLayer):
        return True  # like transformer_auto_wrap_policy
    if isinstance(module, torch.nn.Linear) and all(
        p.requires_grad for p in module.parameters()
    ):
        return True  # wrap each trainable linear separately
    return False


def main():
    args = parse_args()

    if WORLD_SIZE > 1:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    auto_wrap_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=False,
        ignored_modules=None,
        limit_all_gathers=True,
        use_orig_params=True,
        param_init_fn=None,
        sync_module_states=True,
        forward_prefetch=False,
        activation_checkpointing=False,
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    # accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    if WORLD_SIZE > 1:
        args.micro_batch_size = 1
        accelerator.print(
            f"world size = {args.micro_batch_size}. Setting micro_batch_size =1"
        )

    if args.checkpoint_dir is None:
        model_name = args.model_name
    else:
        model_name = args.checkpoint_dir.split("/")[-1]
    # **************************************************************************************
    model = get_model(
        accelerator=accelerator,
        model_name_or_path=args.checkpoint_dir,
        precision=torch.bfloat16,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    is_finetuned = False
    if args.finetuned_path:
        assert args.step is None or args.ckpt_epoch is None
        from peft import PeftModel

        is_finetuned = True
        accelerator.print("loading pretrained peft models")
        if args.ckpt_epoch is not None:
            ckpt = f"epoch_{args.ckpt_epoch}"
            finetune_details = f"{model_name}/{args.dataset_name}/{args.finetuned_mode}/{args.finetuned_epochs}epochs/{ckpt}"
        elif args.step is not None:
            assert args.split == "dev+validation"
            ckpt = f"10ckpts/step_{args.step}"
            finetune_details = f"{model_name}/{args.dataset_name}/{args.finetuned_mode}/{args.finetuned_epochs}epochs/{ckpt}"
        path = f"{args.finetuned_path}/{finetune_details}"
        model = PeftModel.from_pretrained(model, path)
        model.print_trainable_parameters()

    # ***************************************************************************************

    tokenizer = get_tokenizer(
        tokenizer_path=args.tokenizer_dir, model_path=args.checkpoint_dir
    )
    max_seq_len = model.config.max_position_embeddings
    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len
    accelerator.print(max_seq_len)

    # useless in this case:
    pad_token_id = tokenizer.pad_token_id
    accelerator.print("pad_token_id", pad_token_id)
    n_layer = model.config.num_hidden_layers
    accelerator.print("model loaded. \n\n")
    sys.stdout.flush()

    if args.dataset_name == "mmlu":
        dataset_class = mmlu_dataset(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            accelerator=accelerator,
            num_few_shots=args.num_few_shots,
            num_processes=args.preprocessing_num_workers,
            split=args.split,
        )

    elif args.dataset_name == "scienceqa":
        accelerator.print("dataset: scienceqa")
        accelerator.print(f"num_few_shots: {args.num_few_shots}")
        if args.few_shot_topics:
            accelerator.print("subjects = topics")
        else:
            accelerator.print("subjects = category")
        if args.prompt_mmlu:
            accelerator.print("mmlu prompt")

        dataset_class = scienceqa_dataset(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            num_few_shots=args.num_few_shots,
            accelerator=accelerator,
            num_processes=args.preprocessing_num_workers,
            split=args.split,
            prompt_mmlu=args.prompt_mmlu,
            few_shot_topics=args.few_shot_topics,
        )
    elif args.dataset_name == "mmlu_pro_race":
        subject = ["biology", "business"]
        if args.dev_index is not None:
            print("few_shot_index", args.dev_index)
        dataset_class = mmlu_pro_race(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            num_few_shots=args.num_few_shots,
            accelerator=accelerator,
            num_processes=args.preprocessing_num_workers,
            split=args.split,
            subject=subject,
            dev_index=args.dev_index,
        )

    dataset, longest_seq = dataset_class.construct_dataset()

    accelerator.print("num few shots:", args.num_few_shots)
    accelerator.print("max_seq_len:", len(longest_seq["input_ids"][0]))

    dataloader = get_dataloader(
        dataset,
        args.micro_batch_size,
        pad_token_id,
        world_size=WORLD_SIZE,
        shuffle=False,
        num_processes=args.preprocessing_num_workers,
    )

    # ***********************************************************************

    # Put the model on with `accelerator`.
    print_memory_consumed(accelerator.process_index)
    model = accelerator.prepare(model)
    accelerator.print("model loaded to gpus")
    print_memory_consumed(accelerator.process_index)

    # just few forward passes with the longest sequences
    accelerator.print("testing longest seq fints into memory..")
    sys.stdout.flush()
    is_memory_enough(
        model, longest_seq, args.micro_batch_size, pad_token_id, WORLD_SIZE
    )
    print_memory_consumed(accelerator.process_index)
    sys.stdout.flush()

    target_layers = get_target_layers(
        model=model,
        n_layer=n_layer,
        option=args.target_layer,
        every=args.layer_interval,
        world_size=WORLD_SIZE,
        finetuned=is_finetuned,
    )

    nsamples = len(dataloader.dataset)
    accelerator.print("num_total_samples", nsamples)

    inner_path = f"{args.dataset_name}"
    if args.dataset_name == "scienceqa":
        # some opttions for the scienceqa dataset are save in different categories
        if args.few_shot_topics:
            inner_path += "/few_shot_topics/"
        else:
            inner_path += "/few_shot_category/"
        if args.prompt_mmlu:
            inner_path += "/prompt_mmlu/"

    inner_path += f"/evaluated_{args.split}/{model_name}/{args.num_few_shots}shot"

    if args.finetuned_path:
        inner_path = f"finetuned_{args.finetuned_mode}/evaluated_{args.split}/{model_name}/{args.finetuned_epochs}epochs/{ckpt}"

    dirpath = args.out_dir + f"/{inner_path}"
    estract_representations(
        accelerator=accelerator,
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        target_layers=target_layers,
        maxk=args.maxk,
        dirpath=dirpath,
        filename=args.out_filename,
        remove_duplicates=args.remove_duplicates,
        save_distances=args.save_distances,
        save_repr=args.save_repr,
        print_every=args.logging_steps,
    )


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    main()
