## The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Language Models

Source code of the paper:  **The representation landscape of few-shot learning and fine-tuning in large language models** 


### Build an environment with the required dependencies
```
conda create -n repr_fs_ft python=3.11 pip
conda activate repr_fs_ft

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install packaging==24.1
pip install -r requirements.txt
```

### Extract the intermediate representations in a slurm cluster
```
sbatch submit_extract
```

### More to come ...
