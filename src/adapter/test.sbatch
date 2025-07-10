#!/bin/bash
#SBATCH -JAdapterTest
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=8 --mem-per-gpu=32G
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/eval_adapter.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

python src/adapter/test.py --save_path "data/cloaked/cloak_pubfig_small_flat_CosFaceR100_cosine_pgd_cloak_triplet_10_16_multi_adapter" --model_load_path "src/adapter/model_pubfig.pth"