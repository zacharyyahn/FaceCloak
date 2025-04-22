#!/bin/bash
#SBATCH -JFace
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=1 --mem-per-gpu=32G 
#SBATCH -t16:00:00
#SBATCH -ooutput/evaluate_cloak.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

nvidia-smi

DATASET_FILE="data/pubfig_flat"

# NOTE: Only support batch size of 1 right now because of errors with MTCNN breaking when not detecting a face.

python src/cloak.py --dataset_path $DATASET_FILE --batch_size 1
