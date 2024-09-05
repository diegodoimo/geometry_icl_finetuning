#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --account=LADE
#SBATCH --nodes=1
##SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
##SBATCH --begin=now+2hours
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --job-name=icl-vs-sft
#SBATCH --output=output_job/metrics_job_%j.out

# take model name as input
cd $HOME/helm_suite/representation_landscape_fs_ft/notebook/
export PYTHONPATH=/orfeo/cephfs/home/dssc/zenocosini/helm_suite/representation_landscape_fs_ft
model_name="llama-2-70b"
poetry run python plot_script.py --model-name "$model_name" \
                                 --path-ft "/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results/finetuned_dev_val_balanced_40samples/evaluated_test/$model_name/4epochs/epoch_4"

