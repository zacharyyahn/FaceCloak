#!/bin/bash
#SBATCH -JFaceEval
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=4 --mem-per-gpu=80G
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/eval_cloak_fix.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/pubfig_small_flat/"
CLOAK_FILE="data/cloaked/cloak_pubfig_small_flat_Facenet_cosine_pgd_cloak_triplet_8/"
#CLOAK_FILE="data/pubfig_small_flat/"

python src/evaluate.py --dataset_file $DATASET_FILE --probe_file $CLOAK_FILE --num_probes 500 --dataset_size 1.0 --gallery_size 500 --cropped_im_size 112 --verbosity "none" --models "Facenet" --distance_function "cosine"