from collections import defaultdict
import sys
import numpy as np
import pickle
import os
import argparse
from utils import analyze


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer_path",
        type=str,
        default="analysis/test_mask_200.npy",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="analysis/test_mask_200.npy",
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
        "--eval_dataset",
        type=str,
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    return args


# **************************************************************************
args = parse_args()

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
    dataset_mask = np.load(args.mask_path)
else:
    assert False, "dataset misspecified"


overlaps = defaultdict(list)
clusters = defaultdict(list)
intrinsic_dim = defaultdict(list)

args.results_path += f"/statistics/pretrained/{args.model_name}"
os.makedirs(args.results_path, exist_ok=True)

# compute the statistics for all the shots unless a number of shot is given
num_shots = np.arange(6)
if args.num_shots is not None:
    num_shots = [args.num_shots]

for shot in num_shots:
    print("num_shot:", shot)
    sys.stdout.flush()
    name = f"{args.model_name}_mmlu_test_{shot}shot"

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
