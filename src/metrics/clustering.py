from src.utils.annotations import Array, _N_JOBS
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
            with Parallel(n_jobs=_N_JOBS) as parallel:
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
        