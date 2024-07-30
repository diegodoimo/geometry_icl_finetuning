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
from jaxtyping import Float, Int
from typing import Tuple, Optional, LiteralString, List


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
        torch.load(os.path.join(storage_path, file))
        for file in filtered_files
    ]

    # Stack all tensors along a new dimension
    stacked_tensor = torch.stack(tensors[:-1])
    logits = tensors[-1]
    return stacked_tensor.float().cpu(), logits.float().cpu()


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
        ) -> Tuple[Float[Array, "num_instances num_layers d_model"],
                   Float[Array, "num_instances d_vocab"],
                   pd.DataFrame]:
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
    tuple of (np.ndarray, np.ndarray, pd.DataFrame)
        Returns a tuple containing arrays of hidden states, logits, and 
        the original DataFrame.
    """
    storage_path = Path(storage_path)
    # if mask_path:
    #     path_mask = Path(mask_path)

    if not storage_path.exists() or not storage_path.is_dir():
        raise DataRetrievalError(f"Storage path does not exist:"
                                f"{storage_path}")

    files = os.listdir(storage_path)

    # hidden_states, logits = merge_torch_tensors(
    #     storage_path, files
    # )

    mat_dist, md_logits = merge_tensors(
        "dist", storage_path, files
    )
    mat_coord, mc_logits = merge_tensors(
        "index", storage_path, files
    )
    mat_inverse, mi_inverse = merge_tensors(
        "inverse", storage_path, files
    )
    # return (hidden_states, logits), (mat_dist, md_logits), (mat_coord, mc_logits), (mat_inverse, mi_inverse)
    # def rearrange_tensor(tensor, inverse):
    #     new_tensor = np.empty((32, 14042, 51))
    #     for layer in range(mat_dist.shape[0]):
    #         new_tensor[layer] = tensor[layer][inverse[layer]]
    #     return new_tensor
    
    def compute_pull_back(mat_inverse):
        pull_back = []
        for row in mat_inverse:
            _, indices = np.unique(row, return_index=True)
            pull_back.append(indices)
        return pull_back
    
    pull_back = compute_pull_back(mat_inverse)
    
    # retrieve statistics
    with open(Path(storage_path, "statistics_target.pkl"), "rb") as f:
        stat_target = pickle.load(f)

    labels = {"subjects": stat_target["subjects"],
              "predictions": stat_target["contrained_predictions"]}
    labels["subjects"] = preprocess_label(labels["subjects"], pull_back)
    labels["predictions"] = preprocess_label(labels["predictions"], pull_back)
    
    
    if instances_per_sub != -1:
        raise NotImplementedError("This feature is not implemented yet.")
        # mask = sample_indices(labels["subjects"][0], instances_per_sub)
        # mat_dist = mat_dist[:, mask]
        # md_logits = md_logits[mask]
        # mat_coord = mat_coord[:, mask]
        # mc_logits = mc_logits[mask]
        # labels["subjects"] = labels["subjects"][:, mask]
        # labels["predictions"] = labels["predictions"][:, mask]   

    return mat_dist, md_logits, mat_coord, mc_logits, labels
            

def map_label_to_int(my_list: List[str]
                     ) -> Int[Array, "num_layers num_instances"]:
    unique_categories = sorted(list(set(my_list)))
    category_to_int = {category: index
                       for index, category in enumerate(unique_categories)}
    numerical_list = [category_to_int[category] for category in my_list]
    numerical_array = np.array(numerical_list)
    return numerical_array


def preprocess_label(label_array: Int[Array, "num_instances"],
                     pull_back: Int[Array, "num_layers num_instances"]
                     ) -> Int[Array, "num_layers num_instances"]:
    label_array = map_label_to_int(label_array)
    label_array = np.repeat(label_array[np.newaxis, :], 32, axis=0)
    row_indices = np.arange(label_array.shape[0])[:, None]
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