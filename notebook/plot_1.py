import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.metrics.intrinsic_dimension import IntrinsicDimension
from src.metrics.clustering import LabelClustering
from src.metrics.probe import LinearProbe
from src.utils.tensor_storage import retrieve_from_storage, preprocess_label
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plot_config = {
    #'font.size': 12,           
    'axes.titlesize': 30,      
    'axes.labelsize': 29,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 23,
    'figure.figsize': (10,8),
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}


def plotter(data, title, ylabel, yticks = None):
    # Set the style
    sns.set_style(
        "whitegrid",
        rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
    )
    # Setup figure and axes for 2 plots in one row
    plt.figure(dpi = 200)
    layers = np.arange(0,data[0].shape[0])

    #Set ticks
    if layers.shape[0] < 50:
        tick_positions = np.arange(0, layers.shape[0], 4)  # Generates positions 0, 4, 8, ...
    else:
        tick_positions = np.arange(0, layers.shape[0], 8)  # Generates positions 0, 4, 8, ...

    tick_labels = tick_positions +1 # Get the corresponding labels from x

    
    # names = ["5 shot run 1", 
    #         "5 shot run 2", 
    #         "5 shot run 3",
    #         "5 shot run 4",
    #         "5 shot run 5",
    #         "5 shot run shuffled"]
    names = ["0 shot pt",
             "1 shot pt",
             "2 shot pt",
             "5 shot pt",
             "0 shot ft"]
    markerstyle = ['o', 'o', 'o', 'o', 'x']
    
    for int_dim, label, markerstyle in zip(data, names, markerstyle):
        sns.scatterplot(x=layers, y=int_dim, marker=markerstyle)
        sns.lineplot(x=layers, y=int_dim, label=label)


    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    if yticks:
        plt.xticks(ticks=tick_positions, labels=tick_labels)
        tick_positions_y = np.arange(0, yticks, yticks/10).round(2)
        plt.yticks(tick_positions_y)
    plt.tick_params(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.rcParams.update(plot_config)
    plt.savefig(f"plots/{title.replace(' ', '_')}.png")


_PATH = Path("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
            "/repo/results/evaluated_test/random_order/llama-3-70b"
        )
shot = [0,1,2,4]
data_letter = []
for i in shot:
    clustering = LabelClustering()
    out_from_storage = retrieve_from_storage(_PATH / f'{i}shot',
                                            full_tensor=True)
    # import pdb; pdb.set_trace()
    tensors, labels, number_of_layers = out_from_storage
    data_letter.append(clustering.main(z=1.68,
                                tensors=tensors,
                                labels=labels["predictions"],
                                number_of_layers=number_of_layers,
                                parallel=True))
_PATH_ft = Path("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
                "/repo/results/finetuned_dev_val_balanced_40samples"
                "/evaluated_test/llama-3-70b")
clustering = LabelClustering()
out_from_storage = retrieve_from_storage(_PATH_ft,
                                        full_tensor=True,
                                        instances_per_sub=200)
tensors, labels, number_of_layers = out_from_storage
data_letter.append(clustering.main(z=1.68,
                            tensors=tensors,
                            labels=labels["predictions"],
                            number_of_layers=number_of_layers,
                            parallel=True))

ari = [np.array(i['adjusted_rand_score']) for i in data_letter]
plotter(ari, "Label Clustering", "ARI", 0.78)
