# The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Transformer Models

Welcome 👋 to the official repository for the paper **"The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Transformer Models."** This repository contains the source code necessary to reproduce the experiments and analyses presented in the paper.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Extract Intermediate Representations](#extract-intermediate-representations)
  - [Fine-Tuning](#fine-tuning)
  - [Compute Metrics and Plot Results](#compute-metrics-and-plot-results)
- [Running on SLURM Cluster](#running-on-slurm-cluster)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 📖 Overview

This project explores the representation landscape of few-shot learning and fine-tuning in large transformer models. Our goal is to understand how these models adapt and learn with minimal data and how intermediate representations evolve during this process.

Key components of the project include:
- **Representation Extraction**: Scripts to extract intermediate representations from transformer models.
- **Fine-Tuning**: Tools to fine-tune models on custom datasets using various strategies.
- **Analysis and Evaluation**: Notebooks and scripts to analyze and visualize the results.

## 🛠️ Installation

To set up the environment and install the necessary dependencies, we use `poetry`. Follow the steps below to get started.

### Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Build the Environment

1. Clone the repository to your local machine:

    ```
    git clone git@github.com:diegodoimo/geometry_icl_finetuning.git
    cd geometry_icl_finetuning
    ```

2. Install the required dependencies using `poetry`:

    ```
    poetry install
    ```
3. Activate the environment with:
    
    ```
    poetry shell
    ```

This will create a virtual environment and install all necessary packages specified in the `pyproject.toml` file.

## 🚀 Usage

### Extract Intermediate Representations

To extract the intermediate representations from a transformer model, run the following script. This script generates representations for each layer of the model based on the specified configuration.

```
python scripts/extract_repr.py --checkpoint_dir "/path/to/hf/model" or name_of_the_hf_repo
                                --use_slow_tokenizer 
                                --preprocessing_num_workers 16
                                --micro_batch_size 1
                                --out_dir "./results"
                                --logging_steps 100
                                --layer_interval 1
                                --remove_duplicates
                                --use_last_token
                                --max_seq_len 4090
                                --split "test"
                                --num_few_shots 5
                                --dataset_name "name_of_the_dataset"
                                --dataset_path "/path/to/the/dataset"
```

#### Important Arguments

- `--checkpoint_dir`: Name or path of the directory of the Hugging Face model.
- `--dataset_name`: Name of the dataset to use.
- `--dataset_path`: Path to the dataset.

### Fine-Tuning

Fine-tune a transformer model using your custom dataset by running:

```
python scripts/finetune.py --dataset_name name_of_the_dataset
                            --dataset_path /path/to/the/dataset
                            --mask_path /path/to/the/dataset
                            --samples_per_subject 50
                            --model_name_or_path /your/path/to/hf/model
                            --tokenizer_name /your/path/to/hf/model
                            --use_lora
                            --lora_rank 64
                            --lora_alpha 16
                            --lora_dropout 0.1
                            --use_slow_tokenizer
                            --low_cpu_mem_usage
                            --max_seq_length 1024
                            --batch_size 16
                            --preprocessing_num_workers 16
                            --per_device_train_batch_size 1
                            --per_device_eval_batch_size 1
                            --learning_rate 1e-4
                            --warmup_ratio 0.05
                            --weight_decay 0.0
                            --num_train_epochs 1
                            --output_dir /your/path/to/store/the/result/
                            --out_filename ""
                            --checkpointing_steps 10
                            --logging_steps 20
                            --eval_steps 4
                            --weight_samples
                            --use_flash_attn
                            --measure_baselines
```

#### Important Arguments

- `--model_name_or_path`: Hugging Face model path or name of the repo (e.g *meta-llama/LLama-2-7b*).
- `--output_dir`: Directory to save fine-tuning results.
- `--learning_rate`: Learning rate for the optimizer.

### Compute Metrics and Plot Results

Once you have obtained the results from fine-tuning or representation extraction, you can compute metrics and generate plots by executing the provided Jupyter notebook.

To compute the metrics and plot the results, run the following notebook:

```
jupyter notebook notebook/plot.py
```

This notebook contains code to load the results, compute various metrics, and visualize the findings. Ensure that you have Jupyter Notebook installed and set up in your environment.

## Running on SLURM Cluster

To run the extraction script on a SLURM cluster, submit a job using:

```
sbatch scripts/submit_extract
```

Make sure to adjust the SLURM parameters in the `submit_extract` script according to your cluster's configuration.

## 📁 Project Structure

```
geometry_icl_finetuning/
├── README.md                      # Project documentation
├── __init__.py                    # Marks the directory as a Python package
├── assets/                        # Directory for storing assets (e.g., figures, model weights)
├── notebook/                      # Contains notebooks for running experiments with specific datasets
│   └── experiment_notebook.ipynb  # Example notebook for experiments
├── scripts/                       # Scripts to automate tasks like extraction and finetuning
│   ├── extract_repr.py            # Extracts intermediate representations from a dataset
│   ├── finetune.py                # Finetunes a model on a given dataset
│   ├── submit_extract             # SLURM job script for launching `extract_repr.py`
│   └── submit_finetune            # SLURM job script for launching `finetune.py`
├── src/                           # Source code for the project
│   ├── extraction/                # Code related to feature/representation extraction
│   ├── logger/                    # Logging utilities
│   ├── metrics/                   # Metrics and evaluation routines for experiments
│   └── utils/                     # Utility functions for various tasks
└── tests/                         # Unit and integration tests for the project
```

## Contributing

We welcome contributions to this project. If you have an idea or a bug fix, please open an issue or submit a pull request. Make sure to follow our contribution guidelines and code of conduct.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please reach out to the project maintainers:

- [Your Name](mailto:your.email@example.com)
- [Collaborator Name](mailto:collaborator.email@example.com)

We appreciate your interest in our work and hope this repository is helpful for your research and projects!
