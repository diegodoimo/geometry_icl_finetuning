from src.utils.error import DataRetrievalError
from src.utils.annotations import Array
import numpy as np
import torch
import pandas as pd
from einops import rearrange

import os
from pathlib import Path
import re
import pickle
from jaxtyping import Float, Int, Bool
from typing import Tuple, Optional, LiteralString, List, Union, Dict


def merge_torch_tensors(
        storage_path: Path,
        files: List[str]
        ) -> Tuple[Float[Array, "num_layers num_instances d_model"],
                   Float[Array, "num_instances d_vocab"]]:
    pattern_string = f"l(\d+)_target.pt"
    pattern = re.compile(pattern_string)    
    
    filtered_files = [file for file in files if pattern.match(file)]

    # Sort files based on the number in the filename
    filtered_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    # Load tensors and add them to a list
    tensors = [
        torch.load(os.path.join(storage_path, file),
                   weights_only=True)
        for file in filtered_files
    ]

    # Stack all tensors along a new dimension
    stacked_tensor = torch.stack(tensors[:-1])
    logits = tensors[-1]
    return stacked_tensor.float().cpu().numpy()[1:], logits.float().cpu().numpy()[1:]


def merge_tensors(
        type: LiteralString,
        storage_path: Path,
        files: List[str]
        ) -> Tuple[Float[Array, "num_layers num_instances d_model"],
                   Float[Array, "num_instances d_vocab"]]:

    pattern_string = f"l(\d+)_target_{type}\.npy"
    pattern = re.compile(pattern_string)    
    
    filtered_files = [file for file in files if pattern.match(file)]

    # Sort files based on the number in the filename
    filtered_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    # Load tensors and add them to a list
    tensors = [
        np.load(os.path.join(storage_path, file))
        for file in filtered_files
    ]

    # Stack all tensors along a new dimension
    stacked_tensor = np.stack(tensors[:-1])
    logits = tensors[-1]
    return stacked_tensor, logits


def retrieve_from_storage(
        storage_path: Path,
        instances_per_sub: Optional[int] = -1,
        full_tensor: Bool = False,
        ) -> Union[Float[Array, "num_instances num_layers d_model"],
                   Dict[str, Int[Array, "num_layers num_instances"]]] | \
                    Union[Tuple[Float[Array, "num_instances num_layers nearest_neigh"], 
                                Float[Array, "num_instances num_layers d_vocab"]], 
                                Dict[str, Int[Array, "num_layers num_instances"]]]:
    """
    Retrieve tensors from a given path.

    Parameters
    ----------
    storage_path : Path
        Path to the directory containing the tensors.
    instances_per_sub : Optional[int]
        Number of instances per subject in the mmlu dataset.
    mask_path : Optional[Path]
        Path to the mask file for selecting a the requested
        number of instances.
    Returns
    -------
    tuple (np.ndarray, np.ndarray, )
          Returns a tuple containing arrays of hidden states, logits, and 
          .
    """
    storage_path = Path(storage_path)
    # if mask_path:
    #     path_mask = Path(mask_path)

    if not storage_path.exists() or not storage_path.is_dir():
        raise DataRetrievalError(f"Storage path does not exist:"
                                f"{storage_path}")

    if instances_per_sub != -1:
        raise NotImplementedError("This feature is not implemented yet.")

    with open(Path(storage_path, "statistics_target.pkl"), "rb") as f:
        stat_target = pickle.load(f)

    labels = {"subjects": stat_target["subjects"],
            "predictions": stat_target["contrained_predictions"]}
    
    files = os.listdir(storage_path)
    if full_tensor:
        hidden_states, logits = merge_torch_tensors(
            storage_path, files
        )
        num_layers = hidden_states.shape[0]
        labels["subjects"] = preprocess_label(labels["subjects"],
                                              num_layers=num_layers)
        labels["predictions"] = preprocess_label(labels["predictions"],
                                                 num_layers=num_layers)
        return hidden_states, labels, num_layers
    else:
        mat_dist, md_logits = merge_tensors(
            "dist", storage_path, files
        )
        mat_coord, mc_logits = merge_tensors(
            "index", storage_path, files
        )
        mat_inverse, mi_inverse = merge_tensors(
            "inverse", storage_path, files
        )
        def compute_pull_back(mat_inverse):
            pull_back = []
            for row in mat_inverse:
                _, indices = np.unique(row, return_index=True)
                pull_back.append(indices)
            return pull_back
   
        pull_back = compute_pull_back(mat_inverse)
        num_layers = mat_dist.shape[0]
        labels["subjects"] = preprocess_label(labels["subjects"],
                                              pull_back=pull_back,
                                              num_layers=num_layers)
        labels["predictions"] = preprocess_label(labels["predictions"],
                                                 pull_back=pull_back,
                                                 num_layers=num_layers)
        return (mat_dist, md_logits, mat_coord, mc_logits), labels, num_layers
       

def map_label_to_int(my_list: List[str]
                     ) -> Int[Array, "num_layers num_instances"]:
    unique_categories = sorted(list(set(my_list)))
    category_to_int = {category: index
                       for index, category in enumerate(unique_categories)}
    numerical_list = [category_to_int[category] for category in my_list]
    numerical_array = np.array(numerical_list)
    return numerical_array


def preprocess_label(label_array: Int[Array, "num_instances"],
                     num_layers: Int,
                     pull_back: Optional[
                         Int[Array, "num_layers num_instances"]] = None,
                     ) -> Int[Array, "num_layers num_instances"]:
    label_array = map_label_to_int(label_array)
    label_array = np.repeat(label_array[np.newaxis, :], num_layers, axis=0)
    row_indices = np.arange(label_array.shape[0])[:, None]
    if pull_back is not None:
        label_array = label_array[row_indices, pull_back]
    return label_array


def sample_indices(A: Int[Array, "num_instances"],
                   max_samples: Int = 200):
    unique_values, counts = np.unique(A, return_counts=True)
    indices = np.arange(len(A))
    sampled_indices = []

    for value in unique_values:
        value_indices = indices[A == value]
        sample_count = min(counts[value], max_samples)
        sampled_value_indices = np.random.choice(value_indices, sample_count, replace=False)
        sampled_indices.extend(sampled_value_indices)

    return np.array(sampled_indices)