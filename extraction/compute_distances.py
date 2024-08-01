import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from extraction.pairwise_distances import compute_distances
from extraction.extract_activations import extract_activations
from transformers import PreTrainedModel
from accelerate import Accelerator
import pickle
from extraction.helpers import (
    get_embdims,
    measure_performance,
    remove_duplicates_func,
)


@torch.inference_mode()
def estract_representations(
    accelerator: Accelerator,
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer,
    target_layers,
    maxk=50,
    dirpath=".",
    filename="",
    remove_duplicates=True,
    save_distances=True,
    save_repr=False,
    print_every=100,
):
    model = model.eval()

    # create folder
    if accelerator.is_main_process:
        dirpath = str(dirpath).lower()
        os.makedirs(dirpath, exist_ok=True)

    # some postfix
    if filename != "":
        filename = "_" + filename
    filename = f"{filename}_target"

    # target layer name e.g token_embedding, ...
    target_layer_names = list(target_layers.values())
    # target layer number  e.g. 0, 1, 2
    target_layer_labels = list(target_layers.keys())
    accelerator.print("layer_to_extract: ", target_layer_labels)

    # get embedding dimension and their dtypes
    embdims, dtypes = get_embdims(model, dataloader, target_layer_names)

    start = time.time()
    # here we initialize the class
    extr_act = extract_activations(
        accelerator,
        model,
        dataloader,
        target_layer_names,
        embdims,
        dtypes,
        use_last_token=True,
        print_every=print_every,
    )
    # here we extract the activations
    extr_act.extract(dataloader, tokenizer)
    accelerator.print(f"num_tokens: {extr_act.hidden_size/10**3}k")
    accelerator.print((time.time() - start) / 3600, "hours")

    if accelerator.is_main_process:
        # dictionary containing the representation
        act_dict = extr_act.hidden_states
        if save_repr:
            for i, (layer, act) in enumerate(act_dict.items()):
                torch.save(act, f"{dirpath}/l{target_layer_labels[i]}{filename}.pt")

        statistics = measure_performance(extr_act, dataloader, tokenizer, accelerator)

        with open(f"{dirpath}/statistics{filename}.pkl", "wb") as f:
            pickle.dump(statistics, f)

        if save_distances:
            for i, (layer, act) in enumerate(act_dict.items()):
                act = act.to(torch.float64).numpy()

                save_backward_indices = False
                if remove_duplicates:

                    act, save_backward_indices, inverse = remove_duplicates_func(
                        act, accelerator
                    )

                n_samples = act.shape[0]
                if n_samples == 1:
                    accelerator.print(
                        f"{layer} has only one sample: distance matrices will not be computed"
                    )
                else:
                    range_scaling = min(1050, n_samples - 1)
                    maxk = min(maxk, n_samples - 1)

                    start = time.time()
                    distances, dist_index, mus, _ = compute_distances(
                        X=act,
                        n_neighbors=maxk + 1,
                        n_jobs=1,
                        working_memory=2048,
                        range_scaling=range_scaling,
                        argsort=False,
                    )
                    accelerator.print((time.time() - start) / 60, "min")

                    np.save(
                        f"{dirpath}/l{target_layer_labels[i]}{filename}_dist",
                        distances,
                    )
                    np.save(
                        f"{dirpath}/l{target_layer_labels[i]}{filename}_index",
                        dist_index,
                    )
                    if save_backward_indices:
                        np.save(
                            f"{dirpath}/l{target_layer_labels[i]}{filename}_inverse",
                            inverse,
                        )
                    np.save(f"{dirpath}/l{target_layer_labels[i]}{filename}_mus", mus)
