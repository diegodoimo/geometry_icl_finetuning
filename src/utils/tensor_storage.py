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
from typing import Tuple, \
                   Optional, \
                   LiteralString, \
                   List, \
                   Union, \
                   Dict, \
                   Literal 


def load(path: Union[str, Path]
         ) -> Union[np.ndarray, torch.Tensor]:
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".pt":
        return torch.load(path, weights_only=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def stack(tensors: List[Union[np.ndarray, torch.Tensor]],
          axis: int = 0):
    if isinstance(tensors[0], torch.Tensor):
        return torch.stack([t if isinstance(t, torch.Tensor)
                            else torch.from_numpy(t)
                            for t in tensors], dim=axis)
    else:
        return np.stack(tensors, axis=axis)


def concatenate(tensors: List[Union[np.ndarray, torch.Tensor]],
                axis: int = 0):
    if isinstance(tensors[0], torch.Tensor):
        return torch.cat([t if isinstance(t, torch.Tensor)
                          else torch.from_numpy(t)
                          for t in tensors], dim=axis)
    else:
        return np.concatenate(tensors, axis=axis)


def save(data: Union[np.ndarray, torch.Tensor],
         path: Union[str, Path]):
    if path.suffix == ".pt":
        torch.save(data, path)
    elif path.suffix == ".npy":
        np.save(path, data)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def merge_tensors(
        type: Literal["npy", "pt"],
        storage_path: Path,
        files: List[str],
        request: Optional[Literal["dist", "index", "inverse"]] = None,
        ) -> Union[Tuple[Float[torch.Tensor, "num_layers num_instances d_model"],
                   Float[torch.Tensor, "num_instances d_vocab"]],
                   Tuple[Float[Array, "num_layers num_instances d_model"],
                   Float[Array, "num_instances d_vocab"]]]:
    """
    The extraction pipeline for the tensors stored in the storage path.
    """

    if type == "pt":
        pattern_string = f"l(\d+)_target.pt"
    elif type == "npy":
        pattern_string = f"l(\d+)_target_{request}\.npy"
    else:
        raise ValueError(f"Unsupported file format: {type}")
    
    pattern = re.compile(pattern_string)    
    
    filtered_files = [file for file in files if pattern.match(file)]

    # Sort files based on the number in the filename
    filtered_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    # Check if the filea exist
    if len(filtered_files) == 0:
        return None
    
    # Load tensors and add them to a list
    tensors = [
        load(Path(os.path.join(storage_path, file)))
        for file in filtered_files
    ]

    # Stack all tensors along a new dimension
    stacked_tensor = stack(tensors[:-1])
    logits = tensors[-1]
    
    if type == "pt":
        return stacked_tensor.float().cpu().numpy()[1:], \
               logits.float().cpu().numpy()[1:]
    elif type == "npy":
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
    
    if not storage_path.exists() or not storage_path.is_dir():
        raise DataRetrievalError(f"Storage path does not exist:"
                                f"{storage_path}")

    path_stat_target = Path(storage_path, "statistics_target.pkl")
    if not path_stat_target.exists():
        with open(Path(storage_path, "statistics_target_sorted_sample42.pkl"), "rb") as f:
            stat_target = pickle.load(f)    
    else:
        with open(path_stat_target, "rb") as f:
            stat_target = pickle.load(f)

    labels = {"subjects": stat_target["subjects"],
              "predictions": stat_target["contrained_predictions"]}
    
    files = os.listdir(storage_path)
    if full_tensor:
        hidden_states, logits = merge_tensors(

            "pt", storage_path, files
        )
        num_layers = hidden_states.shape[0]
        labels["subjects"] = preprocess_label(labels["subjects"],
                                              num_layers=num_layers)
        labels["predictions"] = preprocess_label(labels["predictions"],
                                                 num_layers=num_layers)
        if instances_per_sub != -1:
            indices = sample_indices(labels["subjects"][0], instances_per_sub)
            hidden_states = hidden_states[:, indices]
            labels["subjects"] = labels["subjects"][:, indices]
            labels["predictions"] = labels["predictions"][:, indices]
        return hidden_states, labels, num_layers
    else:
        mat_dist, md_logits = merge_tensors(
            "npy", storage_path, files, request="dist"
        )
        mat_coord, mc_logits = merge_tensors(
            "npy", storage_path, files, request="index"
        )
        
        inverse_out = merge_tensors(
            "npy", storage_path, files, request="inverse"
        )
        if not inverse_out:
            num_layers = mat_dist.shape[0]
            labels["subjects"] = preprocess_label(labels["subjects"],
                                                  num_layers=num_layers)
            labels["predictions"] = preprocess_label(labels["predictions"],
                                                     num_layers=num_layers)
            return (mat_dist, md_logits, mat_coord, mc_logits), \
                labels, \
                num_layers
        else:
            mat_inverse, mi_inverse = inverse_out    
        
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
            return (mat_dist, md_logits, mat_coord, mc_logits), \
                labels, \
                num_layers
       

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