## The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Language Models

Source code of the paper:  [The representation landscape of few-shot learning and fine-tuning in large language models](https://arxiv.org/abs/2409.03662).
This work has been accepted at the [NeurIPS 2024 conference](https://neurips.cc/).
<br>

## Build an environment with the required dependencies
```
conda create -n repr_fs_ft python=3.11 pip
conda activate repr_fs_ft

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install packaging==24.1
pip install -r requirements.txt
```

<br>

### Extract the intermediate representations in a slurm cluster

Extracting the representations for Llama-3-8b in a 5shot takes about 32 minutes on an A100 GPU and requires roughly 34GB of VRAM.

```
sbatch submit_extract
```
The internal representations of the Llama-3-8b models will be saved in the folder `results/layers/mmlu/pretrained/test/llama-3-8b`. 


### Intrinsic dimension, density peaks, ARIs.

The following script takes as input the path where the representations are stored, for instance `results/layers/mmlu/pretrained/test/llama-3-8b/5shot` for 5 shot representations.

It computes the geometrical properties of the representation landscape described in the papers (intrinsic dimension, clusters, ARIs) and saves the metrics in `results/statistics/pretrained/llama-3-8b`

```
layer_path="results/layers/mmlu/pretrained/test/llama-3-8b/5shot"
python analysis/repr_analysis.py \
    --model_name "llama-3-8b" \
    --results_path "./results" \
    --layer_path $layer_path \
    --mask_path "analysis/test_mask_200.npy" \
    --eval_dataset "test" \
    --num_shots 5  
```


### Plot the figures. 
The script takes as input the path where the statistics are stored (e.g. `results/statistics/pretrained/test/llama-3-8b`). It saves the plots in the directory `figures`.

```
statistics_path="results/statistics/pretrained/llama-3-8b"
python plot.py \
    --figures_dir "figures" \
    --statistics_dir $statistics_path
```
The output is shown below.

<table>
  <tr>
    <td><img src=figures/aris.png width="500"></td>
  </tr>
</table>

<br>

## More to come
We will update the scripts to handle the fine-tuned representations and include other plots soon. 

