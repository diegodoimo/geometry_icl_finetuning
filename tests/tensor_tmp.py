# %%
from src.utils.tensor_storage import retrieve_from_storage
import unittest
import numpy as np
from unittest.mock import Mock
import pandas as pd
from pathlib import Path

# %%
_PATH = Path("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
                     "/repo/results/evaluated_test/random_order/llama-3-8b")
out = retrieve_from_storage(_PATH / '0shot')
# %%
hidden_states, logits = out[0]
mat_dist, md_logits = out[1]
mat_coord, mc_logits = out[2]
mat_inverse, mi_inverse = out[3]
# %%
out = np.unique(hidden_states, return_inverse=True, axis = 1)
# %%
out = np.unique(hidden_states[11], return_inverse=True, axis = 0)
# %%
out = np.unique(mat_inverse[10], return_inverse=True, axis = 1)

# %%
pull_back = []
prova = []
for row in mat_inverse:
    # Extract unique elements from the row
    unique_arr, indices = np.unique(row, return_index=True)
    # Append the unique row to C
    pull_back.append(indices)
    prova.append(unique_arr)


pull_back = np.array(pull_back)
prova = np.array(prova)