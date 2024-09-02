import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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


from src.utils.annotations import Array
from src.utils.error import MetricComputationError, DataRetrievalError
from src.utils.tensor_storage import retrieve_from_storage
import logging
from dadapy.data import Data
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, \
                                    adjusted_mutual_info_score, \
                                    completeness_score, \
                                    homogeneity_score

from sklearn.metrics import f1_score
from scipy.stats import entropy

import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
from pathlib import Path
from jaxtyping import Float, Int, Bool
from typing import Dict, LiteralString, List, Tuple


f1_score_micro = partial(f1_score, average="micro")
_COMPARISON_METRICS = {
    "adjusted_rand_score": adjusted_rand_score,
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "mutual_info_score": mutual_info_score,
    "f1_score": f1_score_micro,
    "completeness_score": completeness_score,
    "homogeneity_score": homogeneity_score,

}

#######################
# CLUSTERING
#######################

class LabelClustering():
    def __init__(self):
        pass
    def main(self,
             z: Float,
             tensors: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             labels: Int[Array, "num_layers num_instances"],
             number_of_layers: Int,
             halo: Bool = False,
             parallel: Bool = True
             ) -> Dict[str, Float[Array, "num_layers num_layers"]]:
        """
        Compute the agreement between the clustering of the hidden states
        and a given set of label.
        Output
        ----------
        Dict[layer: List[Float(num_layers, num_layers)]]
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing label cluster")
        # import pdb; pdb.set_trace()
        try:
            
            output_dict = self.parallel_compute(
                number_of_layers, tensors, labels, z, halo, parallel
            )
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data: {e}"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error occured during computation of metrics: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing clustering: {e}"
            )
            raise e

        return output_dict

    def parallel_compute(
        self, 
        number_of_layers: Int,
        tensors: Float[Array, "num_layers num_instances d_model"] |
        Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
        labels: Int[Array, "num_layers num_instances"],
        z: Float,
        halo: Bool = False,
        parallel: Bool = True
    ) -> Dict[str, Float[Array, "num_layers"]]:
        """
        Compute the overlap between a set of representations and a given label
        using Advanced Peak Clustering.
        M.dErrico, E. Facco, A. Laio, A. Rodriguez, Automatic topography of
        high-dimensional data sets by non-parametric density peak clustering,
        Information Sciences 560 (2021) 476492.
        Inputs:
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
                It can either receive the hidden states or the distance matrices
            labels: Float[Int, "num_instances"]
            z: Array
                merging parameter for the clustering algorithm
            halo: bool
                compute (or not) the halo points

        Returns:
            Dict[str, Float[Array, "num_layers"]]
        """        
        process_layer = partial(
            self.process_layer, tensors=tensors, z=z, halo=halo
        )
        results = []
       
        if parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=8) as parallel:
                results = parallel(
                    delayed(process_layer)(layer,
                                           label=labels[layer])
                    for layer in tqdm.tqdm(range(number_of_layers),
                                           desc="Processing layers")
                )
        else:
            for layer in tqdm.tqdm(range(number_of_layers)):
                results.append(process_layer(layer,
                                             label=labels[layer]))
        
        keys = list(results[0].keys())
        output = {key: [] for key in keys}

        # Merge the results
        for layer_result in results:
            for key in output:
                output[key].append(layer_result[key])
        return output

    def process_layer(
            self, 
            layer: Int,
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            label: Float[Int, "num_instances"],
            z: Array,
            halo: Bool = False,
    ) -> Dict[str, Float[Array, "num_layers"]]:
        """
        Process a single layer.
        Inputs:
            layer: Int
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
                It can either receive the hidden states or the distance matrices
            labels: Float[Int, "num_instances"]
            z: Array
                merging parameter for the clustering algorithm
            halo: bool
                compute (or not) the halo points
        Returns:
            Dict[str, Float[Array, "num_layers"]]
        """
        layer_results = {}
        try:
            # do clustering
            if isinstance(tensors, tuple):
                mat_dist, _, mat_coord, _ = tensors
                data = Data(distances=(mat_dist[layer], mat_coord[layer]))
            elif isinstance(tensors, np.ndarray):
                tensors = tensors[layer]
                base_unique, base_idx, base_inverse = np.unique(
                    tensors, axis=0, return_index=True, return_inverse=True
                )
                indices = np.sort(base_idx)
                base_repr = tensors[indices]
                label = label[indices]

                # do clustering
                data = Data(coordinates=base_repr)
                ids, _, _ = data.return_id_scaling_gride(range_max=100)
                data.set_id(ids[3])
                data.compute_density_kNN(k=16)
        
            halo = True if halo else False
            clusters_assignment = data.compute_clustering_ADP(Z=z, halo=halo)
        except Exception as e:
            raise MetricComputationError(f"Error raised by Dadapy: {e}")
        
        try:
            # Comparison metrics
            for key, func in _COMPARISON_METRICS.items():
                layer_results[key] = func(clusters_assignment, label)
        except Exception as e:
            raise MetricComputationError(f"Error raised by sklearn: {e}")

        layer_results["labels"] = label
        layer_results["cluster_assignment"] = clusters_assignment 
        return layer_results
    
    def compute_additional_metrics(self, clustering_result: List[
        Dict[str, Float[Array, "num_layers"]]]
                                   ) -> pd.DataFrame:
        """
        Compute the following metrics for clustering:
            - Number of clusters
            - Number of assigned points
            - Mean cluster size
            - Variance cluster size
            - Entropy of the distribution of assignments
        Inputs:
            
        Returns:
            pd.DataFrame
        """
        df = pd.DataFrame(clustering_result)
        df_rows = []
        for row in df.iterrows():
            row = row[1]
            cluster_assignment_all = row["cluster_assignment"]
            n_layers = len(cluster_assignment_all)
            num_clusters = np.zeros(n_layers)
            num_assigned_points = np.zeros(n_layers)
            mean_cluster_size = np.zeros(n_layers)
            var_cluster_size = np.zeros(n_layers)
            entropy_values = np.zeros(n_layers)
            fraction_most_represented = np.zeros(n_layers)
            subjects_all = np.array(row["labels"])
            for i in range(n_layers):
                assignments = cluster_assignment_all[i]
                subjects = subjects_all[i]

                # Calculate cluster metrics
                valid_assignments = assignments[assignments != -1]
                unique_clusters, counts = np.unique(valid_assignments,
                                                    return_counts=True)

                num_clusters[i] += len(unique_clusters)
                num_assigned_points[i] += len(valid_assignments) / \
                    len(assignments)
                mean_cluster_size[i] += np.mean(counts)
                var_cluster_size[i] += np.var(counts)
                # Entropy of the distribution of assignments
                # Fraction of the most represented class
                for cluster, count_cluster in zip(unique_clusters, counts):
                    iter_subject = subjects[assignments == cluster]
                    unique_sub, counts_sub = np.unique(iter_subject,
                                                       return_counts=True)
                    entropy_values[i] += entropy(counts_sub)*count_cluster
                    if len(counts) > 0:
                        val = (np.max(counts_sub) /
                               np.sum(counts_sub))*count_cluster
                        fraction_most_represented[i] += val
                    else:
                        fraction_most_represented[i] += 0
                    
                entropy_values[i] = entropy_values[i] / \
                    unique_clusters.shape[0]
                fraction_most_represented[i] = fraction_most_represented[i] / \
                    unique_clusters.shape[0]
                
            df_rows.append([num_clusters, 
                            num_assigned_points, 
                            mean_cluster_size, 
                            var_cluster_size, 
                            entropy_values, 
                            fraction_most_represented])

        df_out = pd.DataFrame(df_rows, columns=["num_clusters", 
                                                "num_assigned_points", 
                                                "mean_cluster_size", 
                                                "var_cluster_size", 
                                                "entropy_values", 
                                                "fraction_most_represented"])
        return df_out


#######################
#  TENSOR RETRIEVAL
#######################

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
        # beacause of inconsinstency in the data
        min_instances = min(hidden_states.shape[1], labels["subjects"].shape[1], labels["predictions"].shape[1])
        labels["subjects"] = labels["subjects"][:, :min_instances]
        labels["predictions"] = labels["predictions"][:, :min_instances]
        hidden_states = hidden_states[:, :min_instances]
        if instances_per_sub != -1:
            # indices = sample_indices(labels["subjects"][0], instances_per_sub)
            indices = np.load("test_mask_200.npy")
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


#######################
#  COMPUTING ARI
#######################

_PATH_ft = Path("/orfeo/cephfs/scratch/area/ddoimo/open"\
                "/geometric_lens/repo/results"\
                "/finetuned_dev_val_balanced_20samples"\
                "/evaluated_test/llama-3-70b/4epochs/epoch_4")
clustering = LabelClustering()
out_from_storage = retrieve_from_storage(_PATH_ft,
                                         full_tensor=True,
                                         instances_per_sub=200,
                                         )
tensors, labels, number_of_layers = out_from_storage
ari = clustering.main(z=1.68,
                      tensors=tensors,
                      labels=labels["predictions"],
                      number_of_layers=number_of_layers,
                      parallel=False)
sns.lineplot(x=np.arange(0, len(ari['adjusted_rand_score'])), y=ari['adjusted_rand_score'])
