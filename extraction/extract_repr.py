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
from utils.helpers import get_target_layers_llama, get_target_layers_mistral
from utils.model_utils import get_model
from utils.dataloader_utils import get_dataloader
from utils.dataset_utils import MMLU_Dataset
from utils.tokenizer_utils import get_tokenizer
from intrinsic_dimension.compute_distances import compute_id
import torch
import os
from utils.helpers import print_memory_consumed, is_memory_enough

from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import numpy as np


import datetime


# # Get the current directory (root directory of the package)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add the parent directory to the Python path
# parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.insert(0, parent_dir)


# from dataset_utils.utils import MMLU_Dataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--num_samples",
        type=int,
        default=None,
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
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--gibberish", action="store_true")
    parser.add_argument("--random_subject", action="store_true")
    parser.add_argument("--wrong_answers", action="store_true")
    parser.add_argument("--sample_questions", action="store_true")
    parser.add_argument("--declarative", action="store_true")
    parser.add_argument("--prompt_search", action="store_true")
    parser.add_argument("--aux_few_shot", action="store_true")
    parser.add_argument("--only_question", action="store_true")
    parser.add_argument("--only_answer", action="store_true")
    parser.add_argument("--skip_answer", action="store_true")
    parser.add_argument("--skip_choices", action="store_true")
    parser.add_argument("--random_order", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    os.environ["ACCELERATE_MIXED_PRECISION"] = args.precision

    # we use fsdp also when world size ==1. accelerate issue in casting
    if int(os.environ["WORLD_SIZE"]) > 0:
        os.environ["ACCELERATE_USE_FSDP"] = "true"

        os.environ["ACCELERATE_USE_FSDP"] = "true"

        os.environ["FSDP_SHRDING_STRATEGY"] = "FULL_SHARD"
        os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
        if args.model_name.startswith("llama"):
            os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "LlamaDecoderLayer"
        elif args.model_name.startswith("mistral"):
            os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "MistralDecoderLayer"

        os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
        os.environ["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"
        os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    accelerator = Accelerator(mixed_precision=args.precision)

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
    world_size = accelerator.num_processes
    if world_size > 1:
        args.micro_batch_size = 1
        accelerator.print(
            f"world size = {args.micro_batch_size}. Setting micro_batch_size =1"
        )

    # if args.checkpoint_dir is not None:
    #    model_name_tmp = args.checkpoint_dir.split("/")[-1]
    #    if model_name_tmp.startswith("llama-2") or model_name_tmp.startswith("llama-3"):
    #        model_name = model_name_tmp

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
            finetune_details = f"{model_name}/{args.finetuned_mode}/{args.finetuned_epochs}epochs/{ckpt}"
        elif args.step is not None:
            assert args.split == "dev+validation"

            ckpt = f"10ckpts/step_{args.step}"
            finetune_details = f"{model_name}/{args.finetuned_mode}/{args.finetuned_epochs}epochs/{ckpt}"
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

    if args.random_subject:
        args.num_few_shots = 5

    dataset, longest_seq = MMLU_Dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        num_few_shots=args.num_few_shots,
        accelerator=accelerator,
        subject=None,
        num_processes=args.preprocessing_num_workers,
        num_samples=args.num_samples,
        split=args.split,
        dummy=args.dummy,
        gibberish=args.gibberish,
        random_subject=args.random_subject,
        wrong_answers=args.wrong_answers,
        sample_questions=args.sample_questions,
        declarative=args.declarative,
        aux_few_shot=args.aux_few_shot,
        only_question=args.only_question,
        only_answer=args.only_answer,
        skip_answer=args.skip_answer,
        skip_choices=args.skip_choices,
        random_order=args.random_order,
    ).construct_dataset()

    # print(dataset[0]["prompt"])
    # print(len(dataset))
    # assert False

    time_stamp = None
    if args.prompt_search:
        # mask = np.load("diego/analysis/test_mask_100.npy")
        # dataset = dataset.select(mask)
        # assert len(dataset) == 5700

        time_stamp = datetime.datetime.now().__str__().split(" ")[1][:8]
        with open(f"prompt_search_{time_stamp}.txt", "w") as f:
            f.write(f"prompt template:\n")
            f.write(f"{dataset[0]['prompt']}\n\n")

    accelerator.print("num few shots:", args.num_few_shots)
    accelerator.print("max_seq_len:", len(longest_seq["input_ids"][0]))

    dataloader = get_dataloader(
        dataset,
        args.micro_batch_size,
        pad_token_id,
        max_seq_len=max_seq_len,
        world_size=world_size,
        shuffle=False,
        num_processes=args.preprocessing_num_workers,
    )

    # ***********************************************************************

    # Put the model on with `accelerator`.
    print_memory_consumed(accelerator.process_index)
    model = accelerator.prepare(model)
    accelerator.print("model put to gpus")

    print_memory_consumed(accelerator.process_index)

    # just few forward passes with the longest sequences
    accelerator.print("testing longest seq fints into memory..")
    sys.stdout.flush()

    is_memory_enough(
        model, longest_seq, args.micro_batch_size, pad_token_id, max_seq_len, world_size
    )
    accelerator.print("done")
    print_memory_consumed(accelerator.process_index)
    sys.stdout.flush()

    if model_name.startswith("llama"):
        target_layers = get_target_layers_llama(
            model=model,
            n_layer=n_layer,
            option=args.target_layer,
            every=args.layer_interval,
            world_size=accelerator.num_processes,
            finetuned=is_finetuned,
        )

    elif model_name.startswith("mistral"):
        target_layers = get_target_layers_mistral(
            model=model,
            n_layer=n_layer,
            option=args.target_layer,
            every=args.layer_interval,
            world_size=accelerator.num_processes,
            finetuned=is_finetuned,
        )

    nsamples = len(dataloader.dataset)
    accelerator.print("num_total_samples", nsamples)

    inner_path = f"evaluated_{args.split}/{model_name}/{args.num_few_shots}shot"
    if args.dummy:
        inner_path = f"evaluated_{args.split}/dummy/{model_name}/5shot"
    elif args.gibberish:
        inner_path = f"evaluated_{args.split}/gibberish/{model_name}/5shot"
    elif args.random_subject:
        inner_path = f"evaluated_{args.split}/random_subject/{model_name}/{args.num_few_shots}shot"
    elif args.wrong_answers:
        inner_path = f"evaluated_{args.split}/wrong_answers/{model_name}/{args.num_few_shots}shot"
    elif args.sample_questions:
        inner_path = f"evaluated_{args.split}/questions_sampled13/{model_name}/{args.num_few_shots}shot"
    elif args.random_order:
        inner_path = (
            f"evaluated_{args.split}/random_order/{model_name}/{args.num_few_shots}shot"
        )
    elif args.declarative:
        inner_path = (
            f"evaluated_{args.split}/declarative/{model_name}/{args.num_few_shots}shot"
        )
    elif args.finetuned_path:
        inner_path = f"finetuned_{args.finetuned_mode}/evaluated_{args.split}/{model_name}/{args.finetuned_epochs}epochs/{ckpt}"

    dirpath = args.out_dir + f"/{inner_path}"
    compute_id(
        accelerator=accelerator,
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        target_layers=target_layers,
        maxk=args.maxk,
        dirpath=dirpath,
        filename=args.out_filename,
        use_last_token=args.use_last_token,
        remove_duplicates=args.remove_duplicates,
        save_distances=args.save_distances,
        save_repr=args.save_repr,
        print_every=args.logging_steps,
        prompt_search=args.prompt_search,
        time_stamp=time_stamp,
    )


if __name__ == "__main__":
    main()
