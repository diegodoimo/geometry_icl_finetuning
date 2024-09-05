# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.metrics.intrinsic_dimension import IntrinsicDimension
from src.metrics.clustering import LabelClustering
from src.utils.tensor_storage import retrieve_from_storage
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
plot_config = {
    'axes.titlesize': 30,      
    'axes.labelsize': 29,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 23,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}
# %%
# argparser = argparse.ArgumentParser()
# argparser.add_argument("--model-name", type=str)
# argparser.add_argument("--path-ft", type=str)
# args = argparser.parse_args()
# model_name = args.model_name

avg = 2
model_name = "mistral-1-7b"

title = model_name
title = title.replace("-", " ")
title = title[0].upper() + title[1:]

# path_ft = args.path_ft
# _PATH_ft = Path(path_ft)

_PATH_ft = Path(f"/orfeo/cephfs/scratch/area/ddoimo/open"\
                f"/geometric_lens/repo/results"\
                f"/finetuned_dev_val_balanced_40samples"\
                f"/evaluated_test/{model_name}/6epochs/epoch_6")

_PATH = Path(f"/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
             f"/repo/results/evaluated_test/random_order/{model_name}")


def find_num_shot(path):
    if "70b" in path:
        return 4
    else:
        return 5


def average_custom_blocks(y, n):
    if n == 0:
        return y
    y_avg = []
    y_avg.append(np.mean(y[0:n]))
    if len(y) > n:
        y_avg.append(np.mean(y[0:n+1]))

    for i in range(1, len(y)-1):
        y_avg.append(np.mean(y[i:n+i+1]))
    assert len(y_avg) == len(y), f"y_avg:{len(y_avg)}, y:{len(y)}"

    return np.array(y_avg)


def plotter(file_name, model, data, title, ylabel, yticks=None, avg=avg):
    # Set the style
    sns.set_style(
        "whitegrid",
        rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
    )
    # Setup figure and axes for 2 plots in one row
    plt.figure(dpi=200)
    layers = np.arange(0, data[0].shape[0])

    # Set ticks
    if layers.shape[0] < 50:
        # Generates positions 0, 4, 8, ...
        tick_positions = np.arange(0, layers.shape[0], 4)  
    else:
        # Generates positions 0, 4, 8, ...
        tick_positions = np.arange(0, layers.shape[0], 8)  

    tick_labels = tick_positions + 1  # Get the corresponding labels from x

    names = ["0 shot pt",
             "1 shot pt",
             "2 shot pt",
             "5 shot pt",
             "0 shot ft"]
    markerstyle = ['o', 'o', 'o', 'o', 'x']
   
    for int_dim, label, markerstyle in zip(data, names, markerstyle):
        int_dim = average_custom_blocks(int_dim, avg)
        sns.scatterplot(x=layers, y=int_dim, marker=markerstyle)
        sns.lineplot(x=layers, y=int_dim, label=label)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if yticks:
        plt.xticks(ticks=tick_positions, labels=tick_labels)
        if isinstance(yticks, list):
            tick_positions_y = np.arange(yticks[0], (yticks[1] + (yticks[1]-yticks[0])/10),
                                         (yticks[1]-yticks[0])/10).round(2)
        else:
            tick_positions_y = np.arange(0, (yticks + yticks/10), yticks/10).round(2)
        plt.yticks(tick_positions_y)
    plt.tick_params(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.rcParams.update(plot_config)

    # setting fi
    file_name = file_name.replace(" ", "_")
    if ylabel == "ARI":
        file_name = file_name + "_1.6"
    # file_name = file_name + "_"+model
    file_name = file_name + "_avg_" + str(avg) if avg > 0 else file_name    
    file_name = file_name + "_no_title" if not title else file_name
    if title:
        path = Path(f"plots/{file_name}.pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        # plt.savefig(f"plots/{file_name}.png")
        plt.savefig(path, format='pdf')
    else:
        path = Path(f"plots/no_title/{file_name}.pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        # plt.savefig(f"plots/no_title/{file_name}.png")
        plt.savefig(path, format='pdf')
    plt.show()


_METRICS = ["intrinsic_dimension",
            "clustering_subject",
            "clustering_letter",
            "clusters_analysis"]

print(f"Start computation for model {model_name}\n for metrics:{_METRICS}")
# %%
##############################
# Intrinsic Dimension
##############################
if "intrinsic_dimension" in _METRICS:
    print("Intrinsic Dimension")

    shot = [0, 1, 2, find_num_shot(str(_PATH))]
    data = []

    cache_path = Path(f"cache/{model_name}/")
    cache_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_path / "intrinsic_dim.pkl"):
        print("Loading from cache")
        with open(cache_path / "intrinsic_dim.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        for i in shot:
            out_from_storage = retrieve_from_storage(_PATH / f'{i}shot',
                                                     full_tensor=True,
                                                     instances_per_sub=200)
            tensors, _, number_of_layers = out_from_storage
            intrinsic_dim = IntrinsicDimension()
            
            data.append(intrinsic_dim.main(tensors, number_of_layers))

        out_from_storage = retrieve_from_storage(_PATH_ft,
                                                 full_tensor=True,
                                                 instances_per_sub=200)
        tensors, _, number_of_layers = out_from_storage
        intrinsic_dim = IntrinsicDimension()
        data.append(intrinsic_dim.main(tensors, number_of_layers))
        with open(cache_path / "intrinsic_dim.pkl", "wb") as f:
            pickle.dump(data, f)
    # Selecting the order of nearest neighbors considered in gride
    data_nn_index = [arr[:, -3] for arr in data]
    plotter(file_name="ID/"+model_name, model=model_name,
            data=data_nn_index, title=title, ylabel="ID", avg=0, yticks=22)
    plotter(file_name="ID/"+model_name, model=model_name,
            data=data_nn_index, title=title, ylabel="ID", avg=avg, yticks=22)
    print("Intrinsic Dimension done")
# %%
##############################
# Clustering  Subject
##############################
if "clustering_subject" in _METRICS:
    print("Clustering Subject")
    shot = [0, 1, 2, find_num_shot(str(_PATH))]
    data_subjects = []

    cache_path = Path(f"cache/{model_name}/")
    cache_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_path / "subject.pkl"):
        print("Loading from cache")
        with open(cache_path / "subject.pkl", "rb") as f:
            data_subjects = pickle.load(f)
    else:
        for i in shot:
            clustering = LabelClustering()
            out_from_storage = retrieve_from_storage(_PATH / f'{i}shot',
                                                     full_tensor=True,
                                                     instances_per_sub=200)
            tensors, labels, number_of_layers = out_from_storage
            data_subjects.append(clustering.main(
                z=1.68,
                tensors=tensors,
                labels=labels["subjects"],
                number_of_layers=number_of_layers
                )
            )
            
        clustering = LabelClustering()
        out_from_storage = retrieve_from_storage(_PATH_ft,
                                                 full_tensor=True,
                                                 instances_per_sub=200)
        tensors, labels, number_of_layers = out_from_storage
        data_subjects.append(clustering.main(z=1.68,
                                             tensors=tensors,
                                             labels=labels["subjects"],
                                             number_of_layers=number_of_layers))

        with open(cache_path / "subject.pkl", "wb") as f:
            pickle.dump(data_subjects, f)

    ari = [np.array(i['adjusted_rand_score']) for i in data_subjects]

    plotter(file_name="ari_subjects/"+model_name, model=model_name, data=ari, title=title,
            ylabel="ARI", avg=0, yticks=0.9)
    plotter(file_name="ari_subjects/"+model_name, model=model_name, data=ari, title=title,
            ylabel="ARI", avg=avg, yticks=0.9)
    print("Clustering Subject done")

# %%
##############################
# Clustering  Letters
##############################
if "clustering_letter" in _METRICS:
    print("Clustering Letters")
    shot = [0, 1, 2, find_num_shot(str(_PATH))]
    data_letter = []

    cache_path = Path(f"cache/{model_name}/")
    cache_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_path / "letter.pkl"):
        print("Loading from cache")
        with open(cache_path / "letter.pkl", "rb") as f:
            data_letter = pickle.load(f)
    else:
        for i in shot:
            clustering = LabelClustering()
            out_from_storage = retrieve_from_storage(_PATH / f'{i}shot',
                                                     full_tensor=True,
                                                     instances_per_sub=200)
            tensors, labels, number_of_layers = out_from_storage
            data_letter.append(clustering.main(
                z=1.68,
                tensors=tensors,
                labels=labels["predictions"],
                number_of_layers=number_of_layers
                )
            )
                
        clustering = LabelClustering()
        out_from_storage = retrieve_from_storage(_PATH_ft,
                                                 full_tensor=True,
                                                 instances_per_sub=200)
        tensors, labels, number_of_layers = out_from_storage
        data_letter.append(clustering.main(
            z=1.68,
            tensors=tensors,
            labels=labels["predictions"],
            number_of_layers=number_of_layers
            )
        )
        
        with open(cache_path / "letter.pkl", "wb") as f:
            pickle.dump(data_letter, f)

    ari = [np.array(i['adjusted_rand_score']) for i in data_letter]
    plotter(file_name="ari_letters/"+model_name, model=model_name, data=ari, title=title,
            ylabel="ARI", yticks=0.5, avg=0)
    plotter(file_name="ari_letters/"+model_name, model=model_name, data=ari, title=title,
            ylabel="ARI", yticks=0.5, avg=avg)
    print("Clustering Letters done")

# %%
##############################
# Clusters analysis
##############################
if "clusters_analysis" in _METRICS:
    print("Clusters analysis")
    shot = [0, 1, 2, find_num_shot(str(_PATH))]
    data_subjects_halo = []
    cache_path = Path(f"cache/{model_name}/")
    cache_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_path / "subject_halo.pkl"):
        print("Loading from cache")
        with open(cache_path / "subject_halo.pkl", "rb") as f:
            data_subjects_halo = pickle.load(f)
    else:
        for i in shot:
            clustering = LabelClustering()
            out_from_storage = retrieve_from_storage(_PATH / f'{i}shot',
                                                     full_tensor=True,
                                                     instances_per_sub=200)
            tensors, labels, number_of_layers = out_from_storage
            data_subjects_halo.append(clustering.main(
                z=1.68,
                tensors=tensors,
                labels=labels["subjects"],
                halo=True,
                number_of_layers=number_of_layers,
                )
            )
        
        with open(cache_path / "subject_halo.pkl", "wb") as f:
            pickle.dump(data_subjects_halo, f)

        clustering = LabelClustering()
        out_from_storage = retrieve_from_storage(_PATH_ft,
                                                 full_tensor=True)
        tensors, labels, number_of_layers = out_from_storage
        data_subjects_halo.append(clustering.main(
            z=1.68,
            tensors=tensors,
            labels=labels["subjects"],
            halo=True,      
            number_of_layers=number_of_layers,
            )
        )
        with open(cache_path / "subject_halo.pkl", "wb") as f:
            pickle.dump(data_subjects_halo, f)
    ari = [np.array(i['adjusted_rand_score']) for i in data_subjects]

    label_clustering = LabelClustering()
    
    # set env variable
    os.environ["OPENBLAS_NUM_THREADS"] = "128"
    metrics_subject = label_clustering.compute_additional_metrics(
        data_subjects_halo
        )

    num_clusters = metrics_subject["num_clusters"].to_list()
    plotter(file_name="number_clusters/"+model_name+"-num-cluster", model=model_name, data=num_clusters,
            title=title, ylabel="Number of Clusters", avg=0, yticks=[15, 80])
    plotter(file_name="number_clusters/"+model_name+"-num-cluster", model=model_name, data=num_clusters,
            title=title, ylabel="Number of Clusters", avg=avg, yticks=[15, 80])

    num_clusters = metrics_subject["num_assigned_points"].to_list()
    plotter(file_name="core_points/"+model_name+"-num-ass-point", model=model_name, data=num_clusters,
            title=title, ylabel="Core Point Fraction", avg=0, yticks=[0.1, 0.65])
    plotter(file_name="core_points/"+model_name+"-num-ass-point", model=model_name, data=num_clusters,
            title=title, ylabel="Core Point Fraction", avg=avg, yticks=[0.1, 0.65])
    print("Clusters analysis done")
