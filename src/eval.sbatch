#!/bin/bash
#SBATCH -JFaceEval
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=4 --mem-per-gpu=80G
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/evaluate_diffam.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/facescrub_small_flat/"
CLOAK_FILE="data/cloaked/test_diffam/"


python src/evaluate.py --dataset_file $DATASET_FILE --probe_file $CLOAK_FILE --num_probes 5000 --dataset_size 1.0 --gallery_size 5000 --cropped_im_size 112 --verbosity "none" --models "Facenet" --distance_function "cosine"
