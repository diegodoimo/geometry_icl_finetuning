#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import datasets
import warnings
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import transformers
import numpy as np
import sys
import time
from collections import defaultdict

from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from functools import partial
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from peft import PeftModel
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# *******************************************************************

from utils.dataloader_utils import get_dataloader, DataCollatorForCausalLMEval
from utils.helpers_finetune import (
    print_memory_consumed,
    save_with_accelerate,
    find_grad_accumulation_steps,
    compute_weighted_ce,
    save_statistics,
)
from utils.dataset_utils import mmlu_dataset
from utils.mmlu_pro_race import mmlu_pro_race
from utils.dataloader_utils import get_dataloader
from utils.tokenizer_utils import get_tokenizer
from utils.optimizer_utils import get_optimizer, get_scheduler
from utils.model_utils import get_model


# *******************************************************************

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mmlu",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default=None,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch_size",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Where to store the final model."
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
        "--checkpointing_steps",
        type=int,
        default=10,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
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
        "--gradient_checkpointing",
        action="store_true",
        help=("Turn on gradient checkpointing. Saves memory but slows training."),
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=-1,
        help="Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).",
    )
    parser.add_argument(
        "--measure_baselines",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--measure_overlap",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--samples_per_subject",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
    )
    parser.add_argument("--clip_grad_thresh", type=float, default=1.0)
    parser.add_argument("--lr_min_fact", type=float, default=0.01)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--weight_samples", action="store_true")
    parser.add_argument("--compute_macro_accuracy", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    gradient_accumulation_steps, args.batch_size = find_grad_accumulation_steps(
        args=args, world_size=WORLD_SIZE
    )

    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    fsdp_plugin = None
    if WORLD_SIZE > 1:
        os.environ["ACCELERATE_USE_FSDP"] = "true"

        def lambda_fn(module: torch.nn.Module):
            if isinstance(module, LlamaDecoderLayer):
                return True  # like transformer_auto_wrap_policy
            if isinstance(module, torch.nn.Linear) and all(
                p.requires_grad for p in module.parameters()
            ):
                return True  # wrap each trainable linear separately
            return False

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

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, fsdp_plugin=fsdp_plugin
    )

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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        args.output_dir += f"{args.dataset_name}/dev_val_balanced"
        if args.samples_per_subject is not None:
            args.output_dir += f"_{args.samples_per_subject}samples"

        args.output_dir += f"/{args.num_train_epochs}epochs"
        if args.seed is not None:
            args.output_dir += f"_seed{args.seed}"
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    world_size = accelerator.num_processes

    # # *******************************************************
    # # Load pretrained model and tokenizer

    model = get_model(
        accelerator=accelerator,
        model_name_or_path=args.model_name_or_path,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        precision=torch.bfloat16,
        use_flash_attention_2=args.use_flash_attn,
        activation_checkpointing=args.activation_checkpointing,
    )

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        if args.resume_from_checkpoint:
            accelerator.print("loading pretrained peft models")
            model = PeftModel.from_pretrained(model, args.resume_from_checkpoint)
        else:
            accelerator.print("Initializing LORA model...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=[
                    "q_proj",
                    "o_proj",
                    "v_proj",
                    "k_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            model = get_peft_model(model, peft_config)

        if RANK == 0:
            model.print_trainable_parameters()

    tokenizer = get_tokenizer(
        tokenizer_path=args.tokenizer_name, model_path=args.model_name_or_path
    )

    # *******************************************************************************
    # # Preprocessing the datasets.
    max_seq_len = 2048
    if args.max_seq_length is not None and args.model_name_or_path is not None:
        max_seq_len = args.max_seq_length
    accelerator.print("max_seq_len: ", max_seq_len)

    accelerator.print("preparing training set")
    dataset_class = mmlu_dataset
    dataset_path = None
    if args.dataset_name == "mmlu_pro_race":
        dataset_class = mmlu_pro_race
        dataset_path = args.dataset_path

    train_dataset, _ = dataset_class(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        accelerator=accelerator,
        num_processes=args.preprocessing_num_workers,
        split="train",
        samples_per_subject=args.samples_per_subject,
        dataset_path=dataset_path,
    ).construct_dataset()
    accelerator.print(f"num_samples = {len(train_dataset)}")

    accelerator.print("preparing validation set")
    val_dataset, _ = dataset_class(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        accelerator=accelerator,
        num_processes=args.preprocessing_num_workers,
        split="validation",
        dataset_path=dataset_path,
    ).construct_dataset()

    accelerator.print("preparing test set")
    test_dataset, _ = dataset_class(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        accelerator=accelerator,
        num_processes=args.preprocessing_num_workers,
        split="test",
        dataset_path=dataset_path,
    ).construct_dataset()

    int_to_subject = {
        i: subject for i, subject in enumerate(np.unique(test_dataset["subjects"]))
    }
    subject_to_int = {subj: i for i, subj in int_to_subject.items()}

    # ********************************************************************************
    # we do not hadle the missing pad tokens in Llamas
    assert args.per_device_train_batch_size == 1
    assert args.per_device_eval_batch_size == 1

    # # DataLoaders creation:
    train_loader, sampler = get_dataloader(
        dataset=train_dataset,
        batch_size=args.per_device_train_batch_size,
        pad_token_id=tokenizer.pad_token_id,
        world_size=world_size,
        shuffle=True,
        num_processes=6,
        weight_samples=args.weight_samples,
        return_sampler=True,
    )

    val_loader = get_dataloader(
        val_dataset,
        args.per_device_eval_batch_size,
        tokenizer.pad_token_id,
        world_size=world_size,
        shuffle=False,
        num_processes=6,
        collate_fn=DataCollatorForCausalLMEval(tokenizer.pad_token_id),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = get_dataloader(
            test_dataset,
            args.per_device_eval_batch_size,
            tokenizer.pad_token_id,
            world_size=world_size,
            shuffle=False,
            num_processes=6,
            collate_fn=DataCollatorForCausalLMEval(tokenizer.pad_token_id),
        )

    # *******************************************************************************

    # Prepare everything with `accelerator` model must be prepared before giving it to the optimizer.
    accelerator.print("memory consumed before loading model")
    print_memory_consumed(rank=RANK)
    sys.stdout.flush()
    model = accelerator.prepare(model)
    accelerator.print("memory consumed after loading model")
    print_memory_consumed(rank=RANK)
    sys.stdout.flush()

    # should be done after wrapping the model in FSDP
    if args.activation_checkpointing:
        accelerator.print("preparing activation checkpointing..")
        sys.stdout.flush()

        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    # **********************************************************************************

    accelerator.print("setup scheduler and optimizer..")
    sys.stdout.flush()
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    optimizer = get_optimizer(model, args)
    lr_scheduler, warmup_steps = get_scheduler(optimizer, args)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # ***********************************************************************************
    filename = ""
    if args.out_filename != "":
        filename = "_" + args.out_filename

    eval_steps, _ = get_cpt_steps(args.eval_steps, args.max_train_steps, logspace=False)
    checkpointing_steps, _ = get_cpt_steps(
        args.checkpointing_steps, args.max_train_steps, logspace=True
    )
    log_steps, log_interval = get_cpt_steps(
        args.logging_steps, args.max_train_steps, logspace=False
    )

    stats = defaultdict()
    stats["num_epochs"] = args.num_train_epochs
    stats["lr"] = args.learning_rate
    stats["batch_size"] = args.batch_size
    stats["weight_decay"] = args.weight_decay
    stats["lora_rank"] = args.lora_rank
    stats["lora_alpha"] = args.lora_rank
    stats["lora_dropout"] = args.lora_dropout

    if args.save_checkpoint:
        output_dir = "epoch_0"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        # save pretrained model
        accelerator.print("saving pretrained model at initialization..")
        sys.stdout.flush()
        save_with_accelerate(accelerator, model, output_dir, args)

    if args.compute_macro_accuracy:
        warnings.warn("computing macro accuracy will slow down the forward pass.")

    train_stats = defaultdict(dict)
    if args.measure_baselines:
        accelerator.print("measuring baselines..")
        sys.stdout.flush()

        acc = evaluate(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            subject_to_int=subject_to_int,
            int_to_subject=int_to_subject,
            compute_macro=args.compute_macro_accuracy,
        )
        print_memory_consumed(rank=RANK)
        logger.info(
            f"baseline mmlu val accuracy: macro {acc['macro']:.4f}, micro {acc['micro']:.4f}"
        )
        save_statistics(
            train_stats=train_stats,
            stats=stats,
            completed_steps=0,
            epoch=0,
            results_dir=args.output_dir,
            filename=filename,
            acc_val=acc,
        )

    accelerator.print("start training")
    accelerator.print("memory before train run")
    sys.stdout.flush()
    print_memory_consumed(rank=RANK)

    # *******************************************************************************

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Weight Decay = {args.weight_decay}")
    logger.info(f"  Lora Rank = {args.lora_rank}")
    logger.info(f"  Lora Alpha = {args.lora_alpha}")
    logger.info(f"  Lora Dropout = {args.lora_dropout}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total batch size (w. parallel, distributed & accumulation) = {args.batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  len_dataloader = {len(train_loader)}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Warmup steps = {warmup_steps}")
    logger.info(f"  Log steps number = {len(log_steps)}")

    completed_steps = 0
    total_loss = 0
    total_time = 0

    for epoch in range(args.num_train_epochs):

        model.train()
        start = time.time()
        # gradient accumulation step may not finish with a proper update at the end of the epoch so we call zero grad here
        optimizer.zero_grad()
        if WORLD_SIZE > 1:
            sampler.set_epoch(epoch)

        for _, batch in enumerate(train_loader):

            with accelerator.accumulate(model):
                batch = {key: val.to("cuda") for key, val in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                loss = compute_weighted_ce(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    weights=batch["weight"],
                    vocab_size=model.config.vocab_size,
                )

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # if (index + 1) % gradient_accumulation_steps == 0:
                completed_steps += 1
                total_time += time.time() - start

                if completed_steps in log_steps:
                    accelerator.print(f"log step: {completed_steps}/{log_steps[-1]}")
                    sys.stdout.flush()

                    if WORLD_SIZE > 1:
                        total_loss = total_loss.reshape(1)
                        dist.all_reduce(total_loss)

                    avg_loss = (
                        total_loss.item()
                        / WORLD_SIZE
                        / gradient_accumulation_steps
                        / log_interval
                    )

                    accelerator.print(
                        f"LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, \
                            Time: {int(total_time//3600)} h {(total_time%3600)/60: .2f} min"
                    )
                    save_statistics(
                        train_stats,
                        stats,
                        completed_steps,
                        epoch + 1,
                        args.output_dir,
                        filename,
                        loss=avg_loss,
                    )
                    total_loss = 0

                if completed_steps in eval_steps:
                    acc = evaluate(
                        model=model,
                        dataloader=val_loader,
                        tokenizer=tokenizer,
                        subject_to_int=subject_to_int,
                        int_to_subject=int_to_subject,
                        compute_macro=args.compute_macro_accuracy,
                    )
                    print_memory_consumed(rank=RANK)
                    logger.info(
                        f"iter {completed_steps}. mmlu val accuracy: micro {acc['micro']:.4f}"
                    )
                    save_statistics(
                        train_stats,
                        stats,
                        completed_steps,
                        epoch + 1,
                        args.output_dir,
                        filename,
                        acc_val=acc,
                    )

                if completed_steps in checkpointing_steps and args.save_checkpoint:
                    accelerator.print("saving checkpoint")
                    sys.stdout.flush()
                    output_dir = (
                        f"{len(checkpointing_steps)}ckpts/step_{completed_steps}"
                    )
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    save_with_accelerate(accelerator, model, output_dir, args)
                    accelerator.print("check saved")

                if completed_steps >= args.max_train_steps:
                    break
                start = time.time()

        print_memory_consumed(rank=RANK)
        # save model
        output_dir = f"epoch_{epoch+1}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)

        acc = evaluate(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer,
            subject_to_int=subject_to_int,
            int_to_subject=int_to_subject,
            compute_macro=args.compute_macro_accuracy,
        )
        logger.info(
            f"iter {completed_steps}. mmlu test accuracy: micro {acc['micro']:.4f}"
        )
        save_statistics(
            train_stats,
            stats,
            completed_steps,
            epoch + 1,
            args.output_dir,
            filename,
            acc_test=acc,
        )
        save_with_accelerate(accelerator, model, output_dir, args)


# FSDP has issues with `inference_mode`
# @torch.inference_mode()
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    tokenizer,
    compute_macro=None,
    subject_to_int=None,
    int_to_subject=None,
):
    model.eval()

    predictions, ground_truths, subjects = [], [], []

    for iter_num, batch in enumerate(dataloader):
        if (iter_num + 1) % int(
            1000 / (dataloader.batch_size * WORLD_SIZE)
        ) == 0 and RANK == 0:
            print(
                f"{iter_num * dataloader.batch_size*WORLD_SIZE+1}/ {len(dataloader.dataset)} inputs processed"
            )
            sys.stdout.flush()

        input_ids, targets, mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        input_ids = input_ids.to("cuda")
        outputs = model(input_ids)
        logits = outputs.logits

        seq_len = torch.sum(mask, dim=1)
        last_logits = logits[torch.arange(logits.shape[0]), seq_len - 1]
        predictions += [torch.argmax(last_logits, dim=-1, keepdims=True)]
        ground_truths += [targets]
        subjects.extend(
            [
                torch.tensor([subject_to_int[subj]]).to("cuda")
                for subj in batch["subjects"]
            ]
        )

    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)
    subjects = torch.cat(subjects)

    if WORLD_SIZE > 1:
        pred_list = [torch.zeros_like(predictions) for _ in range(WORLD_SIZE)]
        gt_list = [torch.zeros_like(ground_truths) for _ in range(WORLD_SIZE)]

        dist.all_gather(pred_list, predictions)
        dist.all_gather(gt_list, ground_truths)

        predictions = torch.cat(pred_list, dim=0).cpu()
        ground_truths = torch.cat(gt_list, dim=0).cpu()

        if compute_macro:
            subject_list = [torch.zeros_like(subjects) for _ in range(WORLD_SIZE)]
            dist.all_gather(subject_list, subjects)
            subjects = torch.cat(subject_list, dim=0)

    ground_truths = np.array([tokenizer.decode(tg).strip() for tg in ground_truths])
    predictions = np.array([tokenizer.decode(pred).strip() for pred in predictions])
    subjects = subjects.cpu().numpy()

    acc_pred = compute_accuracy(predictions, ground_truths, subjects, int_to_subject)

    return acc_pred


def compute_accuracy(predictions, answers, subjects=None, int_to_subject=None):

    accuracy = defaultdict(lambda: "not computed")
    tot_ans = len(predictions)
    num_correct = 0
    for pred, ans in zip(predictions, answers):
        if pred.strip() == ans.strip():
            num_correct += 1
    accuracy["micro"] = num_correct / tot_ans

    if subjects is not None:
        acc_subj = {}
        for subject in np.unique(subjects):
            mask = subject == subjects

            pred_tmp = predictions[mask]
            ans_tmp = answers[mask]

            tot_ans = len(ans_tmp)
            num_correct = 0
            for pred, ans in zip(pred_tmp, ans_tmp):
                if pred.strip() == ans.strip():
                    num_correct += 1
            acc_tmp = num_correct / tot_ans

            acc_subj[int_to_subject[subject]] = acc_tmp

        accuracy["subjects"] = acc_subj
        accuracy["macro"] = np.mean(list(acc_subj.values()))

    return accuracy


def get_cpt_steps(nsteps, max_train_steps, logspace=True):

    if logspace:
        steps = np.unique(
            np.around(np.geomspace(1, max_train_steps, nsteps, endpoint=False)).astype(
                int
            )
        )
        step = None
    else:
        step = max(1, int(np.around(max_train_steps / nsteps)))
        steps = np.arange(0, max_train_steps, step).astype(int)

    return steps, step


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    main()
