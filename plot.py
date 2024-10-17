import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def get_repr_for_test(results_dir, dataset, folder, model, split):

    name = f"cluster_{model}_{dataset}_{split}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        clus_train = pickle.load(f)

    return clus_train


# ***********************************************************************************
plots_dir = "figures"
results_dir = "./results/statistics"
os.makedirs(plots_dir, exist_ok=True)


profiles = {}
for shot in [0, 1, 2, 5]:
    name = f"cluster_llama-3-8b_mmlu_test_{shot}shot.pkl"
    with open(f"{results_dir}/pretrained/llama-3-8b/{name}", "rb") as f:
        clusters = pickle.load(f)
    profiles[f"subjects-ari_{shot}shot"] = clusters[f"subjects-ari-{shot}shot-z1.6-k16"]
    profiles[f"letters-ari_{shot}shot"] = clusters[f"letters-ari-{shot}shot-z1.6-k16"]
    profiles[f"core-point-fraction_{shot}shot"] = clusters[
        f"core_point_fraction-{shot}shot"
    ]

sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)
fig = plt.figure(figsize=(8, 3.7))

gs1 = GridSpec(1, 2)
ax = fig.add_subplot(gs1[0])
for shot in [0, 1, 2, 5]:
    sns.lineplot(
        profiles[f"subjects-ari_{shot}shot"],
        marker=".",
        label=f"{shot}shot",
    )
ax.legend()
ax.set_ylabel("ARI subjetcs")
ax.set_xlabel("layers")
ax.set_title("llama-3-8b")
ax.set_xticks(np.arange(0, 33, 4))
ax.set_xticklabels(np.arange(1, 34, 4))
ax.set_ylim(-0.05, 0.9)

ax = fig.add_subplot(gs1[1])
for shot in [0, 1, 2, 5]:
    sns.lineplot(
        profiles[f"letters-ari_{shot}shot"],
        marker=".",
        label=f"{shot}shot",
    )
ax.legend()
ax.set_ylabel("ARI letters")
ax.set_xlabel("layers")
ax.set_title("llama-3-8b")
ax.set_xticks(np.arange(0, 33, 4))
ax.set_xticklabels(np.arange(1, 34, 4))
ax.set_ylim(-0.05, 0.9)

gs1.tight_layout(fig)
plt.savefig(f"{plots_dir}/aris.png", dpi=200)
