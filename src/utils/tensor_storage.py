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
from jaxtyping import Float
from typing import Tuple, Optional, LiteralString, List


def merge_tensors(
        type: LiteralString,
        storage_path: Path,
        files: List[str]
        ) -> Tuple[Float[Array, "num_instances num_layers d_model"],
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
        instances_per_sub: Optional[int] = None,
        mask_path: Optional[Path] = None
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
    # retrieve statistics
    with open(Path(storage_path, "statistics_target.pkl"), "rb") as f:
        stat_target = pickle.load(f)

    # Cropping tensors because of inconsistency in the data
    for key in stat_target.keys():
        value = stat_target.get(key, None)
        if value is None or isinstance(value, float):
            continue 
        elif "accuracy" in key:
            stat_target[key] = stat_target[key]["macro"]
            continue

        # import pdb; pdb.set_trace()
        stat_target[key] = stat_target[key][:14040]

    df = pd.DataFrame(stat_target)
    df = df.rename(
        columns={
            "subjects": "dataset",
            "predictions": "std_pred",
            "answers": "letter_gold",
            "contrained_predictions": "only_ref_pred", # the typo is in the data
        }
    )
    
    mat_dist, md_logits = merge_tensors(
        "dist", storage_path, files
    )
    mat_coord, mc_logits = merge_tensors(
        "coord", storage_path, files
    )
    return mat_dist, md_logits, mat_coord, mc_logits, df
            

# if instances_per_sub:
#         if not mask_path:
#             raise DataRetrievalError("Mask path is not provided")
#         if not path_mask.exists():
#             raise DataRetrievalError(f"Mask path does not exist: "
#                                     f"{path_mask}")

#         mask = np.load(path_mask)
#         stacked_tensor = stacked_tensor.float().numpy()[:14040][mask]
#         logits = logits.float().numpy()[:14040][mask]
#         df = df.iloc[mask]
#         df.reset_index(inplace=True, drop=True)

#     else:
#         stacked_tensor = stacked_tensor.float().numpy()[:14040]
#         logits = logits.float().numpy()[:14040]
#         df = df.iloc[:14040]
#         df.reset_index(inplace=True, drop=True)