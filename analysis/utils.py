from collections import Counter
from dadapy._cython import cython_overlap as c_ov
import numpy as np
import warnings
import torch
import pickle
from sklearn.metrics import adjusted_rand_score
from dadapy import data
from collections import Counter
import math


def return_data_overlap(indices_base, indices_other, k=30, subjects=None):

    assert indices_base.shape[0] == indices_other.shape[0]
    ndata = indices_base.shape[0]

    overlaps_full = c_ov._compute_data_overlap(
        ndata, k, indices_base.astype(int), indices_other.astype(int)
    )

    overlaps = np.mean(overlaps_full)
    if subjects is not None:
        overlaps = {}
        for subject in np.unique(subjects):
            mask = subject == subjects
            overlaps[subject] = np.mean(overlaps_full[mask])

    return overlaps


def _label_imbalance_helper(labels, k, class_fraction):
    if k is not None:
        max_k = k
        k_per_sample = np.array([k for _ in range(len(labels))])

    k_per_class = {}
    class_count = Counter(labels)
    # potentially overwrites k_per_sample
    if class_fraction is not None:
        for label, count in class_count.items():
            class_k = int(count * class_fraction)
            k_per_class[label] = class_k
            if class_k == 0:
                k_per_class[label] = 1
                warnings.warn(
                    f" max_k < 1 for label {label}. max_k set to 1.\
                    Consider increasing class_fraction.",
                    stacklevel=2,
                )
        max_k = max([k for k in k_per_class.values()])
        k_per_sample = np.array([k_per_class[label] for label in labels])

    class_weights = {label: 1 / count for label, count in class_count.items()}
    sample_weights = np.array([class_weights[label] for label in labels])

    return k_per_sample, sample_weights, max_k


def return_label_overlap(
    dist_indices,
    labels,
    k=None,
    avg=True,
    class_fraction=None,
    weighted=True,
):
    k_per_sample, sample_weights, max_k = _label_imbalance_helper(
        labels, k, class_fraction
    )
    # print(k_per_sample, sample_weights, max_k)
    assert len(labels) == dist_indices.shape[0]

    neighbor_index = dist_indices[:, 1 : max_k + 1]

    ground_truth_labels = np.repeat(np.array([labels]).T, repeats=max_k, axis=1)

    overlaps = np.equal(np.array(labels)[neighbor_index], ground_truth_labels)

    if class_fraction is not None:
        nearest_neighbor_rank = np.arange(max_k)[np.newaxis, :]
        # should this overlap entry be discarded?
        mask = nearest_neighbor_rank >= k_per_sample[:, np.newaxis]
        # mask out the entries to be discarded
        overlaps[mask] = False

    overlaps = overlaps.sum(axis=1) / k_per_sample
    if avg and weighted:
        overlaps = np.average(overlaps, weights=sample_weights)
    elif avg:
        overlaps = np.mean(overlaps)

    return overlaps


def analyze(
    base_path,
    layer,
    dataset_mask,
    clusters,
    intrinsic_dim,
    overlaps,
    spec="",
    k=16,
    z=1.6,
    class_fraction=0.3,
):

    activations = torch.load(f"{base_path}/l{layer}_target.pt")
    activations = activations.to(torch.float64).numpy()
    print("dataset_size:", activations.shape[0])

    with open(f"{base_path}/statistics_target.pkl", "rb") as f:
        stats = pickle.load(f)

    subjects = stats["subjects"]
    subjects_to_int = {sub: i for i, sub in enumerate(np.unique(subjects))}
    subj_label = np.array([subjects_to_int[sub] for sub in subjects])

    letters = stats["answers"]
    letters_to_int = {letter: i for i, letter in enumerate(np.unique(letters))}
    letter_label = np.array([letters_to_int[sub] for sub in letters])

    # select up to 200 points per class.
    activations = activations[dataset_mask]
    subj_label = subj_label[dataset_mask]
    letter_label = letter_label[dataset_mask]

    # remove identical points
    _, base_idx, _ = np.unique(
        activations, axis=0, return_index=True, return_inverse=True
    )
    indices = np.sort(base_idx)
    activations = activations[indices]
    subj_label = subj_label[indices]
    letter_label = letter_label[indices]
    print("dataset_size after mask and prune:", activations.shape[0])

    # ***********************************************************************

    maxk = 1000
    assert indices.shape[0] > maxk, (indices.shape[0], maxk)
    # distances_base, dist_index_base, _, _ = compute_distances(
    #     X=activations,
    #     n_neighbors=maxk + 1,
    #     n_jobs=1,
    #     working_memory=2048,
    #     range_scaling=2048,
    #     argsort=False,
    # )

    d = data.Data(coordinates=activations)
    ids, _, _ = d.return_id_scaling_gride(range_max=maxk)
    dist_index_base = d.dist_indices

    # this sets the kNN order of the density estimator consistent with
    # the one used to compute the ID with gride
    id_index = int(math.log2(k))

    d.set_id(ids[id_index])
    intrinsic_dim[f"ids-{spec}"].append(ids)
    d.compute_density_kNN(k=k)
    assignment = d.compute_clustering_ADP(Z=z, halo=False)

    # number of clusters found
    clusters[f"nclus-{spec}-z{z}-k{k}"] = d.N_clusters

    # ARI with subjects
    clusters[f"subjects-ari-{spec}-z{z}-k{k}"] = adjusted_rand_score(
        assignment, subj_label
    )

    # ARI with answers
    clusters[f"letters-ari-{spec}-z{z}-k{k}"] = adjusted_rand_score(
        assignment, letter_label
    )

    # halo computation
    assignment_core = d.compute_clustering_ADP(Z=z, halo=True)
    clusters[f"core_point_fraction-{spec}"] = np.sum(assignment_core != -1) / len(
        assignment_core
    )

    # overlap subjects
    overlaps[f"subjects-{spec}-{class_fraction}"] = return_label_overlap(
        dist_indices=dist_index_base,
        labels=list(subj_label),
        class_fraction=class_fraction,
    )

    # overlap letters
    overlaps[f"letters-{spec}-{class_fraction}"] = return_label_overlap(
        dist_indices=dist_index_base,
        labels=list(letter_label),
        class_fraction=class_fraction,
    )

    return clusters, intrinsic_dim, overlaps
