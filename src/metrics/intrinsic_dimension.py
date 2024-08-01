from src.utils.annotations import Array
from src.utils.error import MetricComputationError, DataRetrievalError
from src.utils.tensor_storage import retrieve_from_storage

from dadapy.data import Data

import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int, Bool
from typing import Tuple


class IntrinsicDimension():
    def __init__(self):
        pass
    def main(self,
             tensors: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "order_of_nearest_neighbour num_layers"]:
        """
        Compute the intrinsic dimension of the hidden states of a model
        Returns
            Float[Array, "order_of_nearest_neighbour num_layers"]
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info("Computing ID")

        try:
            id_per_layer_gride = (
                self.parallel_compute(number_of_layers, tensors, parallel)
            )
        except DataRetrievalError as e:
            module_logger.error(
                f"Error retrieving data. Error: {e}"
            )
            raise
        except MetricComputationError as e:
            module_logger.error(
                f"Error raised by Dadapy. Error: {e}"
            )
            raise
        except Exception as e:
            module_logger.error(
                f"Error computing ID. Error: {e}"
            )
            raise

        return id_per_layer_gride

    def parallel_compute(
            self, 
            number_of_layers: Int,
            tensors: Float[Array, "num_layers num_instances d_model"] | 
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            parallel: Bool = True
            
    ) -> Float[Array, "order_of_nearest_neighbour num_layers"]:
        """
        Collect hidden states of all instances and compute ID using gride
        estimator
        Denti, F., Doimo, D., Laio, A., Mira, A., The generalized ratios intrinsic dimension estimator,
        <<SCIENTIFIC REPORTS>>, 2022; 12 (1): 20005-20020.
        [doi:10.1038/s41598-022-20991-1] [https://hdl.handle.net/10807/219504]
        
        Inputs
            mat_dist: Float[Array, "num_layers num_instances nearest_neigh"]
                Array with the distances of the nearest neighbours
            mat_coord: Float[Array, "num_layers num_instances nearest_neigh"]
                Array with the coordinates of the nearest neighbours
        Returns
            Float[Array, "order of nearest neighbour, num_layers"]
                Array with the ID of each layer,
                for each order of nearest neighbour
        """

        id_per_layer_gride = []
        process_layer = partial(
            self.process_layer, tensors=tensors
        )
        if parallel:
            # Parallelize the computation of the metric
            # If the program crash try reducing the number of jobs
            with Parallel(n_jobs=-1) as parallel:
                id_per_layer_gride = parallel(
                    delayed(process_layer)(layer)
                    for layer in tqdm.tqdm(range(1, number_of_layers),
                                           desc="Processing layers")
                )
        else:
            # Sequential version
            for layer in tqdm.tqdm(range(1, number_of_layers),
                                   desc="Processing layers"):
                id_per_layer_gride.append(process_layer(layer))
        
        # Inserting ID 0 because Dadapy raise an error
        id_per_layer_gride.insert(0, np.ones(id_per_layer_gride[-1].shape[0]))
        return np.stack(id_per_layer_gride)
               
    def process_layer(
            self, 
            layer: Int,
            tensors: Float[Array, "num_layers num_instances d_model"] |
            Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
    ) -> Float[Array, "order_of_nearest neighbour"]:
        """
        Process a single layer
        Inputs
            mat_dist: Float[Array, "num_instances nearest_neigh"]
                Array with the distances of the nearest neighbours
            mat_coord: Float[Array, "num_instances nearest_neigh"]
                Array with the coordinates of the nearest neighbours
        Returns
        """
        try:
            if isinstance(tensors, tuple):
                mat_dist, _, mat_coord, _ = tensors
                data = Data(distances=(mat_dist[layer], mat_coord[layer]))
            elif isinstance(tensors, np.ndarray):
                tensors = tensors[layer]
                # do clustering
                data = Data(coordinates=tensors)
            
            # data.remove_identical_points()
            out = data.return_id_scaling_gride(range_max=1000)[0]
        except Exception as e:
            raise MetricComputationError(f"Error raised by Dadapy: {e}")
        return out
        