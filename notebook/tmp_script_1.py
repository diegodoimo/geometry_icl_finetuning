import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
from pathlib import Path
import seaborn as sns
plot_config = {
    #'font.size': 12,           
    'axes.titlesize': 30,      
    'axes.labelsize': 29,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 23,
    'figure.figsize': (10,8),
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}
from matplotlib import pyplot as plt


import numpy as np
import torch

import os
from pathlib import Path
import pickle

from dadapy.data import Data
from sklearn.metrics.cluster import adjusted_rand_score

import pandas as pd
import numpy as np
from pathlib import Path


def map_label_to_int(my_list):
    unique_categories = sorted(list(set(my_list)))
    category_to_int = {category: index
                       for index, category in enumerate(unique_categories)}
    numerical_list = [category_to_int[category] for category in my_list]
    numerical_array = np.array(numerical_list)
    return numerical_array


def preprocess_label(label_array,
                     num_layers,
                    ):
    label_array = map_label_to_int(label_array)
    
    
    return label_array


def sample_indices(A,
                   max_samples):
    unique_values, counts = np.unique(A, return_counts=True)
    indices = np.arange(len(A))
    sampled_indices = []

    for value in unique_values:
        value_indices = indices[A == value]
        sample_count = min(counts[value], max_samples)
        sampled_value_indices = value_indices[:sample_count] # np.random.choice(value_indices, sample_count, replace=False)
        sampled_indices.extend(sampled_value_indices)

    return np.array(sampled_indices)


def retrieve_from_storage(
        path,
        instances_per_sub,
        layer,
        ):
    
    path_hidden_states = Path(path / f"l{layer}_target.pt")
    
    path_stat_target = Path(path, "statistics_target.pkl")
    
    with open(path_stat_target, "rb") as f:
        stat_target = pickle.load(f)

    labels = {"subjects": stat_target["subjects"],
              # "predictions": stat_target["contrained_predictions"]}
              "predictions": stat_target["answers"]}
    
    hidden_states = torch.load(path_hidden_states, weights_only=True)
    num_layers = hidden_states.shape[0]
    # import pdb; pdb.set_trace()
    labels["subjects"] = preprocess_label(labels["subjects"],
                                          num_layers=num_layers)
    labels["predictions"] = preprocess_label(labels["predictions"],
                                            num_layers=num_layers)
    # beacause of inconsinstency in the data
    
    min_instances = min(hidden_states.shape[0],
                        labels["subjects"].shape[0],
                        labels["predictions"].shape[0])

    labels["subjects"] = labels["subjects"][:min_instances]
    labels["predictions"] = labels["predictions"][:min_instances]
    hidden_states = hidden_states[:min_instances]
    
    if instances_per_sub != -1:
        # indices = sample_indices(labels["subjects"], instances_per_sub)
        # np.save("test_mask_200.npy", indices)
        # indices = np.load("/u/dssc/zenocosini/helm_suite/representation_landscape_fs_ft/test_mask_200.npy")
        indices = np.load("/u/dssc/zenocosini/helm_suite/representation_landscape_fs_ft/assets/test_mask_200.npy")
        hidden_states = hidden_states[indices]
        labels["subjects"] = labels["subjects"][indices]
        labels["predictions"] = labels["predictions"][indices]
    return hidden_states.float().cpu().numpy(), labels, num_layers


def compute_ari(z, tensors, labels):
    
    base_unique, base_idx, base_inverse = np.unique(
        tensors, axis=0, return_index=True, return_inverse=True
    )
    indices = np.sort(base_idx)
    base_repr = tensors[indices]
    labels = labels[indices]

    # do clustering
    data = Data(coordinates=base_repr)
    ids, _, _ = data.return_id_scaling_gride(range_max=100)
    data.set_id(ids[3])
    data.compute_density_kNN(k=16)
    clusters_assignment = data.compute_clustering_ADP(Z=z, halo=False)
    # import pdb; pdb.set_trace()
    ari = adjusted_rand_score(labels, clusters_assignment)
    return ari
    

# _PATH_ft = Path("/orfeo/cephfs/scratch/area/ddoimo/open"\
#                 "/geometric_lens/repo/results"\
#                 "/finetuned_dev_val_balanced_20samples"\
#                 "/evaluated_test/llama-3-70b/4epochs/epoch_4")
_PATH_ft = Path("/orfeo/cephfs/scratch/area/ddoimo/open"\
                "/geometric_lens/repo/results"\
                "/finetuned_dev_val_balanced_20samples"\
                "/evaluated_test/llama-3-8b/4epochs/epoch_4")

model_name = "llama-3-70b"
_PATH = Path(f"/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
            f"/repo/results/evaluated_test/random_order/{model_name}/4shot")
# results = []
# for layer in range(80):
#     out_from_storage = retrieve_from_storage(_PATH_ft,
#                                             instances_per_sub=200,
#                                             layer=layer,
#                                             )
#     tensors, labels, number_of_layers = out_from_storage
#     ari = compute_ari(z=1.68,
#                     tensors=tensors,
#                     labels=labels["predictions"]
#                     )
#     print(f"ARI: {ari}")
#     results.append(ari)
# sns.lineplot(x=range(80), y=results)
# # save the plot
# plt.savefig("ari_plot.png")

layer = 58
out_from_storage = retrieve_from_storage(_PATH,
                                            instances_per_sub=200,
                                            layer=layer,
                                            )
tensors, labels, number_of_layers = out_from_storage
ari = compute_ari(z=1.68,
                tensors=tensors,
                labels=labels["predictions"]
                )
print(f"ARI: {ari}")
