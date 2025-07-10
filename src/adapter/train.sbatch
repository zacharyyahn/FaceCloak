#!/bin/bash
#SBATCH -JAdapter
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=8 --mem-per-gpu=32G
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/test_mem.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

python src/adapter/train.py \
    --lr 0.0001 \
    --epochs 10 \
    --model_save_path "src/adapter/model_pubfig.pth" \
    --dataset_path "data/masks/" \
    #--load_model_from "src/adapter/model_mem.pth"