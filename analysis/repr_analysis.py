from collections import defaultdict
import torch
from dadapy._cython import cython_overlap as c_ov
from pairwise_distances import compute_distances
import sys
import numpy as np
import pickle

import os
import argparse
from dadapy import data
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from utils import return_label_overlap, analyze


rng = np.random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--finetuned_mode",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pretrained_mode",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
    )
    parser.add_argument(
        "--samples_subject",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--do_with_steps",
        action="store_true",
    )
    args = parser.parse_args()
    return args


# **************************************************************************
args = parse_args()

assert args.finetuned_mode is None or args.pretrained_mode is None

# base path
base_dir = "/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results"

if args.model_name in ["llama-3-8b", "mistral-1-7b"]:
    nlayers = 34
elif args.model_name == "llama-2-13b":
    nlayers = 42
elif "70" in args.model_name:
    nlayers = 82
else:
    assert (
        False
    ), f"wrong model name {args.model_name}, expected llama-3-8b or llama-2-13b"

print("analyzing model:", args.model_name)


# ********************************************************************************

# dataset to analyze
if args.eval_dataset == "test":
    dataset_mask = np.load(f"./analysis/test_mask_200.npy")
    is_balanced = f"_balanced200"
elif args.eval_dataset == "dev+validation":
    mask_dir = args.mask_dir
    dataset_mask = np.load(f"./analysis/dev+validation_mask_20.npy")
else:
    assert False, "dataset misspecified"


overlaps = defaultdict(list)
clusters = defaultdict(list)
intrinsic_dim = defaultdict(list)


if args.finetuned_mode is not None:
    print("finetuned_mode:", args.finetuned_mode)
    args.results_path += f"/finetuned/{args.model_name}"
    os.makedirs(args.results_path, exist_ok=True)

    if args.do_with_steps:
        ckpts = [1, 2, 3, 6, 12, 22, 42, 77, 144, 268]
        for i_step, step in enumerate(ckpts[::-1]):
            print("step:", step, f"{i_step+1}/{len(ckpts)}")
            sys.stdout.flush()

            base_path = f"{base_dir}/finetuned_{args.finetuned_mode}/evaluated_{args.eval_dataset}/{args.model_name}/{args.epochs}epochs/10ckpts/step_{step}"
            name = f"{args.model_name}_finetuned_{args.finetuned_mode}_epoch{args.epochs}_10ckpts_eval_{args.eval_dataset}{is_balanced}_0shot"

            for layer in range(1, nlayers):
                print("layer:", f"{layer}/{nlayers-1}")
                sys.stdout.flush()

                clusters, intrinsic_dim, overlaps = analyze(
                    base_path,
                    layer,
                    dataset_mask,
                    clusters,
                    intrinsic_dim,
                    overlaps,
                    spec=f"step-{step}",
                )
    else:
        ckpts = np.arange(args.epochs + 1)
        if args.ckpt is not None:
            ckpts = [args.ckpt]

        for epoch in ckpts[::-1]:
            base_path = f"{base_dir}/finetuned_{args.finetuned_mode}/evaluated_{args.eval_dataset}/{args.model_name}/{args.epochs}epochs/epoch_{epoch}"
            name = f"{args.model_name}_finetuned_{args.finetuned_mode}_epoch{args.epochs}_eval_{args.eval_dataset}{is_balanced}_0shot"

            print("epoch:", epoch)
            sys.stdout.flush()

            for layer in range(1, nlayers):
                print("layer:", f"{layer}/{nlayers-1}")
                sys.stdout.flush()

                clusters, intrinsic_dim, overlaps = analyze(
                    base_path,
                    layer,
                    dataset_mask,
                    clusters,
                    intrinsic_dim,
                    overlaps,
                    spec=f"ep-{epoch}",
                )

            with open(f"{args.results_path}/overlap_{name}.pkl", "wb") as f:
                pickle.dump(overlaps, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f"{args.results_path}/cluster_{name}.pkl", "wb") as f:
                sys.stdout.flush()
                pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f"{args.results_path}/ids_{name}.pkl", "wb") as f:
                pickle.dump(intrinsic_dim, f, protocol=pickle.HIGHEST_PROTOCOL)

elif args.pretrained_mode is not None:
    print("pretrained mode:", args.pretrained_mode)
    args.results_path += f"/pretrained/{args.model_name}"
    os.makedirs(args.results_path, exist_ok=True)
    num_shots = np.arange(6)
    if args.num_shots is not None:
        num_shots = [args.num_shots]

    for shot in num_shots[::-1]:
        # base_path = f"{base_dir}/evaluated_{args.eval_dataset}/{args.pretrained_mode}/{args.model_name}/{shot}shot"
        name = f"{args.model_name}_eval_{args.eval_dataset}{is_balanced}_{shot}shot"
        print("num_shot:", shot)
        sys.stdout.flush()

        for layer in range(1, nlayers):
            print("layer:", f"{layer}/{nlayers-1}")
            sys.stdout.flush()

            clusters, intrinsic_dim, overlaps = analyze(
                args.layer_path,
                layer,
                dataset_mask,
                clusters,
                intrinsic_dim,
                overlaps,
                spec=f"{shot}shot",
            )

        with open(f"{args.results_path}/overlap_{name}.pkl", "wb") as f:
            pickle.dump(overlaps, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{args.results_path}/cluster_{name}.pkl", "wb") as f:
            sys.stdout.flush()
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{args.results_path}/ids_{name}.pkl", "wb") as f:
            pickle.dump(intrinsic_dim, f, protocol=pickle.HIGHEST_PROTOCOL)
