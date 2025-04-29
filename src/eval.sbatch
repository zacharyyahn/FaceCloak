#!/bin/bash
#SBATCH -JFaceEval
#SBATCH -N1 --gres=gpu:1 --ntasks-per-node=4 --mem-per-gpu=32G
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/evaluate_preprocess.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/pubfig_flat/"
CLOAK_FILE="data/pubfig_flat/"


python src/evaluate.py --dataset_file $DATASET_FILE --probe_file $CLOAK_FILE --num_probes 250 --dataset_size 1.0 --gallery_size 100 150 200 300 400 500 1000 2000 5000 10000 --cropped_im_size 112 --verbosity "none" --models "Facenet"
