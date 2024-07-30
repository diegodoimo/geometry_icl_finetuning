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
from typing import Dict, LiteralString, List


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
    def __init__(self,
                 path: Path,
                 parallel: Bool = True):
        self.path = path
        self.parallel = parallel
    
    def main(self,
             z: Float,
             label: LiteralString,
             halo: Bool = False, 
             instance_per_sub: int = -1) -> pd.DataFrame:
        """
        Compute the overlap between the layers of instances in which the model
        answered with the same letter
        Output
        ----------
        Dict[layer: List[Float(num_layers, num_layers)]]
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing label cluster with label {label}")

        self.label = label
        
        tensor_tuple = retrieve_from_storage(self.path, instance_per_sub)
        mat_dist, _, mat_coord, _, labels = tensor_tuple
        label_per_row = labels[self.label]
        try:
            output_dict = self.parallel_compute(
                mat_dist, mat_coord, label_per_row, z, halo
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
        mat_dist: Float[Array, "num_layers num_instances nearest_neigh"],
        mat_coord: Float[Array, "num_layers num_instances nearest_neigh"],
        label: Int[Array, "num_layers num_instances"],
        z: Float,
        halo: Bool = False,
    ) -> Dict[str, Float[Array, "num_layers"]]:
        """
        Compute the overlap between a set of representations and a given label
        using Advanced Peak Clustering.
        M.dErrico, E. Facco, A. Laio, A. Rodriguez, Automatic topography of
        high-dimensional data sets by non-parametric density peak clustering,
        Information Sciences 560 (2021) 476492.
        Inputs:
            mat_dist: Float[Array, "num_layers num_instances nearest_neigh"]
                    Array with the distances of the nearest neighbours
            mat_coord: Float[Array, "num_layers num_instances nearest_neigh"]
                    Array with the coordinates of the nearest neighbours  
            labels: Float[Int, "num_instances"]
            z: Array
                merging parameter for the clustering algorithm
            halo: bool
                compute (or not) the halo points

        Returns:
            Dict[str, Float[Array, "num_layers"]]
        """
        assert (
            mat_dist.shape[1] == label.shape[1]
        ), "Label lenght don't match the number of instances"
        number_of_layers = mat_dist.shape[0]
        
        process_layer = partial(
            self.process_layer, z=z, halo=halo
        )
        results = []
        if self.parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=-1) as parallel:
                results = parallel(
                    delayed(process_layer)(mat_dist[layer],
                                           mat_coord[layer],
                                           label=label[layer])
                    for layer in tqdm.tqdm(range(number_of_layers),
                                           desc="Processing layers")
                )
        else:
            for layer in tqdm.tqdm(range(number_of_layers)):
                results.append(process_layer(mat_dist[layer],
                                             mat_coord[layer],
                                             label=label[layer]))
        
        keys = list(results[0].keys())
        output = {key: [] for key in keys}

        # Merge the results
        for layer_result in results:
            for key in output:
                output[key].append(layer_result[key])
        return output

    def process_layer(
            self, 
            mat_dist: Float[Array, "num_instances nearest_neigh"],
            mat_coord: Float[Array, "num_instances nearest_neigh"],
            label: Float[Int, "num_instances"],
            z: Array,
            halo: Bool = False,
    ) -> Dict[str, Float[Array, "num_layers"]]:
        """
        Process a single layer.
        Inputs:
            layer: Int
            hidden_states: Float[Array, "num_instances, num_layers, model_dim"]
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
            data = Data(distances=(mat_dist, mat_coord))
            ids, _, _ = data.return_id_scaling_gride(range_max=50)
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
                unique_clusters, counts = np.unique(valid_assignments, return_counts=True)

                num_clusters[i] += len(unique_clusters)
                num_assigned_points[i] += len(valid_assignments)/len(assignments)
                mean_cluster_size[i] += np.mean(counts)
                var_cluster_size[i] += np.var(counts)
                # Entropy of the distribution of assignments
                # Fraction of the most represented class
                for cluster, count_cluster in zip(unique_clusters, counts):
                    iter_subject = subjects[assignments == cluster]
                    unique_sub, counts_sub = np.unique(iter_subject, return_counts=True)
                    entropy_values[i] += entropy(counts_sub)*count_cluster
                    if len(counts) > 0:
                        fraction_most_represented[i] += (np.max(counts_sub) / np.sum(counts_sub))*count_cluster
                    else:
                        fraction_most_represented[i] += 0
                    
                entropy_values[i] = entropy_values[i] / unique_clusters.shape[0]
                fraction_most_represented[i] =  fraction_most_represented[i] / unique_clusters.shape[0]
                
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
        