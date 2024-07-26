from src.utils.annotations import Array
from src.utils.error import MetricComputationError, DataRetrievalError
import logging
from dadapy.data import Data
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, \
                                    adjusted_mutual_info_score, \
                                    completeness_score, \
                                    homogeneity_score

from sklearn.metrics import f1_score

import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
from pathlib import Path
from jaxtyping import Float, Int, Bool
from typing import Dict, LiteralString


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
             halo: Bool = False, ) -> pd.DataFrame:
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
        tsm = self.tensor_storage
        
        tensor_tuple = tsm.retrieve_tensor(self.path)
        mat_dist, _, mat_coord, _, hidden_states_df = tensor_tuple
        label_per_row = self.constructing_labels(
            hidden_states_df, mat_dist
        )
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

    def constructing_labels(
        self, 
        hidden_states_df: pd.DataFrame, 
        mat_dist: Float[Array, "num_layers num_instances nearest_neigh"],
    ) -> Float[Array, "num_instances"]:
        """
        Map the labels to integers and return the labels for each instance 
        in the hidden states.
        Inputs:
            hidden_states_df: pd.DataFrame
            hidden_states: Float[Array, "num_instances, num_layers, model_dim"]
        Returns:
            Float[Int, "num_instances"]
        """
        labels_literals = hidden_states_df[self.label].unique()
        labels_literals.sort()

        map_labels = {class_name: n 
                      for n, class_name in enumerate(labels_literals)}

        label_per_row = hidden_states_df[self.label].reset_index(drop=True)
        label_per_row = np.array(
            [map_labels[class_name] for class_name in label_per_row]
        )[: mat_dist.shape[1]]

        return label_per_row

    def parallel_compute(
        self, 
        mat_dist: Float[Array, "num_layers num_instances nearest_neigh"],
        mat_coord: Float[Array, "num_layers num_instances nearest_neigh"],
        label: Int[Array, "num_instances"],
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
            mat_dist.shape[1] == label.shape[0]
        ), "Label lenght don't mactch the number of instances"
        number_of_layers = mat_dist.shape[0]
        
        process_layer = partial(
            self.process_layer, label=label, z=z, halo=halo
        )
        results = []
        if self.parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=-1) as parallel:
                results = parallel(
                    delayed(process_layer)(mat_dist[layer], mat_coord[layer])
                    for layer in tqdm.tqdm(range(1, number_of_layers),
                                           desc="Processing layers")
                )
        else:
            for layer in tqdm.tqdm(range(1, number_of_layers)):
                results.append(process_layer(mat_dist[layer], mat_coord[layer]))
        
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
            data = Data(coordinates=mat_coord, distances=mat_dist)
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
        return layer_results