#!/bin/bash
#SBATCH -JFaceEval
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=4 --mem-per-gpu=32G
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/evaluate_facescrub_benign_facenet.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/facescrub_flat/"
CLOAK_FILE="data/facescrub_flat/"


python src/evaluate.py --dataset_file $DATASET_FILE --probe_file $CLOAK_FILE --num_probes 500 --dataset_size 1.0 --gallery_size 5 10 15 20 25 50 75 100 200 300 400 500 1000 2000 5000 10000 --cropped_im_size 112 --verbosity "none" --models "Facenet" --distance_function "cosine"
