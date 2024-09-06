from src.utils.annotations import Array, _N_JOBS
from src.utils.error import MetricComputationError, DataRetrievalError
import logging
from dadapy.data import Data
import tqdm
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int, Bool
from typing import Tuple



class PointOverlap():
    def __init__(self):
        pass
        
    def main(self,
             k: Int,
             input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             input_j: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
             number_of_layers: Int,
             parallel: Bool = True
             ) -> Float[Array, "num_layers"]:
        """
        Compute overlap between two sets of representations.

        Returns:
            pd.DataFrame
                Dataframe containing results
        """
        module_logger = logging.getLogger(__name__)
        module_logger.info(f"Computing point overlap")


        try:
            overlap = self.parallel_compute(input_i=input_i,
                                            input_j=input_j,
                                            k=k,
                                            number_of_layers=number_of_layers,
                                            parallel=parallel)
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

        return overlap

    

    def parallel_compute(
            self,
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j:  Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            number_of_layers: Int, 
            k: Int,
            parallel: Bool = True
        ) -> Float[Array, "num_layers"]:
        """
        Compute the overlap between two set of representations for each layer.

        Inputs:
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Float[Array, "num_layers"]
        """
        assert (
            input_i.shape[1] == input_j.shape[1]
        ), "The two runs must have the same number of layers"
        process_layer = partial(self.process_layer,
                                input_i=input_i,
                                input_j=input_j,
                                k=k)

        if parallel:
            with Parallel(n_jobs=_N_JOBS) as parallel:
                results = parallel(
                    delayed(process_layer)(layer)
                    for layer in tqdm.tqdm(
                        range(number_of_layers), desc="Processing layers"
                    )
                )
        else:
            results = []
            for layer in range(number_of_layers):
                results.append(process_layer(layer))

        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(
            self,
            layer: Int,
            input_i: Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            input_j:  Float[Array, "num_layers num_instances d_model"] |
                Tuple[Float[Array, "num_layers num_instances nearest_neigh"]],
            k: Int
        ) -> Float[Array, "num_layers"]:
        """
        Process a single layer
        Inputs:
            layer: Int
            input_i: Float[Array, "num_layers, num_instances, model_dim"]
            input_j: Float[Array, "num_layers, num_instances, model_dim"]
            k: Int
                the number of neighbours considered for the overlap
        Returns:
            Array
        """

        input_i = input_i[layer, :, :]
        input_j = input_j[layer, :, :]

        data = Data(coordinates=input_i, maxk=k)
        overlap = data.return_data_overlap(input_j, k=k)
        return overlap

