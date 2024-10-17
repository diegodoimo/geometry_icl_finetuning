import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def get_repr_for_test(results_dir, dataset, folder, model, split):

    name = f"cluster_{model}_{dataset}_{split}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        clus_train = pickle.load(f)

    return clus_train


# ***********************************************************************************
plots_dir = "results/plots"
results_dir = "results/statistics"


# clus_test_ll3_ft, ov_test_ll3_ft = get_repr_for_test(
#     results_dir=results_dir,
#     folder="finetuned",
#     model="llama-3-8b",
#     split="test",
# )

# clus_test_ll3_pt, ov_test_ll3_pt = get_repr_for_test(
#     results_dir=results_dir,
#     folder="pretrained",
#     model="llama-3-8b",
#     dataset="mmlu",
#     split="test",
# )


name = f"cluster_llama-3-8b_mmlu_test_0shot.pkl"
with open(f"{results_dir}/pretrained/llama-3-8b/{name}", "rb") as f:
    clus_train = pickle.load(f)


clus_train

# fig = plt.figure(figsize=(13, 3.7))
fig = plt.figure(figsize=(6, 3.7))

gs1 = GridSpec(1, 1)
ax = fig.add_subplot(gs1[0])
# sns.lineplot(
#     clus_test_ll3_ft["letters-ari-ep-4-z1.6"],
#     marker=".",
#     label="fine-tuned",
# )

sns.lineplot(
    clus_test_ll3_pt["letters-ari-5shot-z1.6"],
    marker=".",
    label="5shot",
)
ax.legend()
ax.set_ylabel("ARI answers")
ax.set_xlabel("layers")
ax.set_title("llama-3-8b")
ax.set_xticks(np.arange(1, 32, 4))
ax.set_xticklabels(np.arange(1, 32, 4))

gs1.tight_layout(fig)
plt.savefig(f"{plots_dir}/overlap_labels.png", dpi=200)
